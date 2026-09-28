// grpc_client_cuda.cu — Flow B2: C++ gRPC -> Triton with CUDA shared memory
//
// Memory chain (zero-copy):
//   NVDEC AV_PIX_FMT_CUDA frames (GPU)
//   -> fused kernel writes letterboxed tensor DIRECTLY into a cudaIpc-registered
//      region -> Triton reads it (server-side H2D eliminated)
//   -> server writes result into cudaIpc output region
//   -> compact kernel reads output on GPU, only candidate boxes cross to host
//
// Modes: rtsp (live) | file (capacity replay of preprocessed frames)
#include <cuda_runtime_api.h>
#define TRITON_ENABLE_GPU 1
#include <grpc_client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace tc = triton::client;

#define CUDA_CHECK(x)                                                          \
  do { cudaError_t e = (x); if (e != cudaSuccess) {                             \
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " @" << __LINE__ << std::endl; exit(1); } \
  } while (0)
#define CHECK_OK(x)                                                            \
  do { tc::Error err = (x); if (!err.IsOk()) {                                  \
    std::cerr << "client error: " << err << std::endl; exit(1); } } while (0)

// ---------------- kernels (shared with flow A2) ----------------
__global__ void nv12_letterbox_kernel(
    const unsigned char* __restrict__ src_y,
    const unsigned char* __restrict__ src_uv,
    int y_pitch, int uv_pitch, int src_w, int src_h,
    float* __restrict__ dst, int nw, int nh, int pad_x, int pad_y, int IMG) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= IMG || y >= IMG) return;
  int sx = x - pad_x, sy = y - pad_y;
  bool inside = (sx >= 0 && sx < nw && sy >= 0 && sy < nh);
  float r, g, b;
  if (!inside) { r = g = b = 114.0f / 255.0f; }
  else {
    // Bilinear luma, nearest chroma. Center-aligned sampling, matching
    // cv2 INTER_LINEAR: src = (dst + 0.5) * scale - 0.5. Nearest-neighbour here
    // measured -1.25% mAP50-95 against the reference (see docs/accuracy.md);
    // chroma stays nearest because NV12 already subsamples it 2x.
    float fx = ((float)sx + 0.5f) * (float)src_w / (float)nw - 0.5f;
    float fy = ((float)sy + 0.5f) * (float)src_h / (float)nh - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - (float)x0, ay = fy - (float)y0;
    int x0c = min(max(x0, 0), src_w - 1), x1c = min(max(x0 + 1, 0), src_w - 1);
    int y0c = min(max(y0, 0), src_h - 1), y1c = min(max(y0 + 1, 0), src_h - 1);
    float Y00 = (float)src_y[(size_t)y0c * y_pitch + x0c];
    float Y01 = (float)src_y[(size_t)y0c * y_pitch + x1c];
    float Y10 = (float)src_y[(size_t)y1c * y_pitch + x0c];
    float Y11 = (float)src_y[(size_t)y1c * y_pitch + x1c];
    float Yb = (Y00 * (1.0f - ax) + Y01 * ax) * (1.0f - ay)
             + (Y10 * (1.0f - ax) + Y11 * ax) * ay;
    int u = min(max((int)(fx + 0.5f), 0), src_w - 1);
    int v = min(max((int)(fy + 0.5f), 0), src_h - 1);
    unsigned char U = src_uv[(size_t)(v >> 1) * uv_pitch + ((u >> 1) << 1)];
    unsigned char V = src_uv[(size_t)(v >> 1) * uv_pitch + ((u >> 1) << 1) + 1];
    float yf = Yb - 16.0f, uf = (float)U - 128.0f, vf = (float)V - 128.0f;
    r = fmaxf(0.f, fminf(1.164f * yf + 1.596f * vf, 255.f)) / 255.0f;
    g = fmaxf(0.f, fminf(1.164f * yf - 0.392f * uf - 0.813f * vf, 255.f)) / 255.0f;
    b = fmaxf(0.f, fminf(1.164f * yf + 2.017f * uf, 255.f)) / 255.0f;
  }
  size_t plane = (size_t)IMG * IMG;
  dst[0 * plane + y * IMG + x] = r;
  dst[1 * plane + y * IMG + x] = g;
  dst[2 * plane + y * IMG + x] = b;
}

struct GPUDet { float x1, y1, x2, y2, score; int cls; };

struct B2Stages { double grpc = 0, post = 0, nms = 0; long n = 0; std::mutex mtx; };
static B2Stages g_b2_stages;

__global__ void compact_candidates_kernel(
    const float* __restrict__ out, int num_classes, int num_anchors, float conf_thr,
    GPUDet* __restrict__ dets, int* __restrict__ d_count, int max_dets) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= num_anchors) return;
  float best = 0.f; int best_c = -1;
  for (int c = 0; c < num_classes; ++c) {
    float s = out[(4 + c) * num_anchors + a];
    if (s > best) { best = s; best_c = c; }
  }
  if (best < conf_thr) return;
  float cx = out[0 * num_anchors + a], cy = out[1 * num_anchors + a];
  float w = out[2 * num_anchors + a], h = out[3 * num_anchors + a];
  int slot = atomicAdd(d_count, 1);
  if (slot < max_dets) dets[slot] = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, best_c};
}

static int nms_count(const GPUDet* dets, int n, float iou_thr) {
  std::vector<GPUDet> v(dets, dets + n);
  std::sort(v.begin(), v.end(), [](const GPUDet& a, const GPUDet& b) { return a.score > b.score; });
  std::vector<bool> removed(v.size(), false);
  int kept = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (removed[i]) continue;
    kept++;
    for (size_t j = i + 1; j < v.size(); ++j) {
      if (removed[j] || v[i].cls != v[j].cls) continue;
      float ix1 = std::max(v[i].x1, v[j].x1), iy1 = std::max(v[i].y1, v[j].y1);
      float ix2 = std::min(v[i].x2, v[j].x2), iy2 = std::min(v[i].y2, v[j].y2);
      float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
      float uni = (v[i].x2 - v[i].x1) * (v[i].y2 - v[i].y1) +
                  (v[j].x2 - v[j].x1) * (v[j].y2 - v[j].y1) - inter;
      if (inter / std::max(uni, 1e-6f) > iou_thr) removed[j] = true;
    }
  }
  return kept;
}

// frames.bin, loaded ONCE in main() and shared read-only by every stream. Each
// stream used to read its own copy: 1.47 GB x 16 streams = 23.5 GB of host RAM,
// which just fit; at 32 streams the copies alone exceed RAM and the kernel kills
// the client (exit 137) before it prints anything.
static std::vector<char> g_frame_buf;

struct StreamCtx {
  std::string url, model, mode, file_path;
  int stream_id, streams;
  std::atomic<long>* frames;
  std::atomic<long>* dets;
  std::vector<double>* latencies;
  std::mutex* lat_mtx;
  double duration;
  // --mode paced: this stream is one open-loop virtual camera
  double fps = 0;
  std::string phase = "random";
  unsigned seed = 1;
  double warmup = 1.0;   // seconds of frames sent but not recorded
  std::chrono::steady_clock::time_point t_start{};
  std::atomic<long>* late = nullptr;
};


// Unregisters a client's CUDA shared-memory regions when the stream's scope
// ends, on every path out of it. Left registered, Triton keeps each run's GPU
// buffers mapped (CUDA IPC) after this process exits, and they pile up across
// a sweep's runs (827 regions / +2.6 GB after 19 runs in the companion study).
struct ShmRegions {
  tc::InferenceServerGrpcClient* c;
  std::vector<std::string> names;
  ~ShmRegions() { for (auto& n : names) c->UnregisterCudaSharedMemory(n); }
};

static void run_stream(StreamCtx* ctx) {
  const int IMG = 640, NUM_CLASSES = 80, NUM_ANCHORS = 8400, MAX_DETS = 4096;
  const size_t IN_BYTES = (size_t)IMG * IMG * 3 * sizeof(float);
  const size_t OUT_BYTES = (size_t)84 * NUM_ANCHORS * sizeof(float);

  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  CHECK_OK(tc::InferenceServerGrpcClient::Create(&client, "localhost:8001", false));

  // ---- register CUDA shm regions (input written by our kernel, output by server) ----
  float* shm_in = nullptr; float* shm_out = nullptr;
  CUDA_CHECK(cudaMalloc(&shm_in, IN_BYTES));
  CUDA_CHECK(cudaMalloc(&shm_out, OUT_BYTES));
  cudaIpcMemHandle_t in_handle, out_handle;
  CUDA_CHECK(cudaIpcGetMemHandle(&in_handle, shm_in));
  CUDA_CHECK(cudaIpcGetMemHandle(&out_handle, shm_out));
  std::string in_region = "in_" + std::to_string(ctx->stream_id) + "_" + std::to_string(getpid());
  std::string out_region = "out_" + std::to_string(ctx->stream_id) + "_" + std::to_string(getpid());
  CHECK_OK(client->RegisterCudaSharedMemory(in_region, in_handle, 0, IN_BYTES));
  CHECK_OK(client->RegisterCudaSharedMemory(out_region, out_handle, 0, OUT_BYTES));
  ShmRegions shm_regions{client.get(), {in_region, out_region}};  // unregistered on every return

  std::vector<int64_t> in_shape = {1, 3, IMG, IMG};
  tc::InferInput* inp = nullptr;
  CHECK_OK(tc::InferInput::Create(&inp, "images", in_shape, "FP32"));
  CHECK_OK(inp->SetSharedMemory(in_region, IN_BYTES, 0));
  tc::InferRequestedOutput* out = nullptr;
  CHECK_OK(tc::InferRequestedOutput::Create(&out, "output0"));
  CHECK_OK(out->SetSharedMemory(out_region, OUT_BYTES, 0));
  std::vector<tc::InferInput*> inputs = {inp};
  std::vector<const tc::InferRequestedOutput*> outputs = {out};

  // output-side compact scratch
  GPUDet* d_dets; int* d_count;
  CUDA_CHECK(cudaMalloc(&d_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  GPUDet* h_dets; int* h_count;
  CUDA_CHECK(cudaMallocHost((void**)&h_dets, MAX_DETS * sizeof(GPUDet)));
  CUDA_CHECK(cudaMallocHost((void**)&h_count, sizeof(int)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  tc::InferOptions options(ctx->model);
  tc::InferResult* res = nullptr;

  auto do_infer_and_post = [&](long& fi_count) -> double {
    auto ts = std::chrono::steady_clock::now();
    CHECK_OK(client->Infer(&res, options, inputs, outputs));
    auto t1 = std::chrono::steady_clock::now();
    // compact candidates on GPU reading the shm output region, then tiny D2H
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
    compact_candidates_kernel<<<(NUM_ANCHORS + 255) / 256, 256, 0, stream>>>(
        shm_out, NUM_CLASSES, NUM_ANCHORS, 0.25f, d_dets, d_count, MAX_DETS);
    CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int n = std::min(*h_count, MAX_DETS);
    CUDA_CHECK(cudaMemcpy(h_dets, d_dets, n * sizeof(GPUDet), cudaMemcpyDeviceToHost));
    auto t2 = std::chrono::steady_clock::now();
    double grpc_ms = std::chrono::duration<double, std::milli>(t1 - ts).count();
    double post_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double nms_ms = 0;
    {
      auto t3 = std::chrono::steady_clock::now();
      int kept = nms_count(h_dets, n, 0.45f);
      nms_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t3).count();
      ctx->dets->fetch_add(kept);
    }
    { std::lock_guard<std::mutex> lk(g_b2_stages.mtx);
      g_b2_stages.grpc += grpc_ms; g_b2_stages.post += post_ms; g_b2_stages.nms += nms_ms; g_b2_stages.n++; }
    ctx->frames->fetch_add(1);
    fi_count++;
    delete res; res = nullptr;
    return grpc_ms + post_ms + nms_ms;
  };

  if (ctx->mode == "file" || ctx->mode == "paced") {
    const std::vector<char>& buf = g_frame_buf;
    const size_t n_frames = buf.size() / IN_BYTES;
    if (n_frames == 0) { std::cerr << "no frames in " << ctx->file_path << std::endl; return; }
    long fi = 0;
    if (ctx->mode == "paced") {
      // ---- paced mode: this thread is ONE open-loop virtual camera ----
      // Frame k is due at t_start + offset + k/fps whether or not the server kept
      // up - a camera does not wait for the answer before exposing the next frame.
      // Turnaround is measured from that DUE time, not from when the request left,
      // so a slow server cannot hide its own delay by making the client send later
      // (coordinated omission). It includes the upload into the shm region,
      // because a real camera's frame has to get there too.
      //
      // Capacity mode (--mode file) answers "how much can it process". This
      // answers "how long does a live frame wait", which capacity mode cannot:
      // there the client holds requests in flight and the queue is its own.
      using clk = std::chrono::steady_clock;
      const double period = 1.0 / ctx->fps;
      double offset = 0.0;
      if (ctx->phase == "random") {
        // Seeded per camera, so every server config is tested against the SAME
        // camera phases and arms differ only in the config.
        std::mt19937 rng(ctx->seed * 1000003u + (unsigned)ctx->stream_id);
        offset = std::uniform_real_distribution<double>(0.0, period)(rng);
      }  // "sync": every camera fires together - the worst-case burst
      fi = ctx->stream_id % (long)n_frames;
      std::this_thread::sleep_until(ctx->t_start);
      // Under overload a camera falls ever further behind schedule. Stop draining
      // that backlog a few seconds after the window so an overloaded run ends.
      const auto hard_stop = ctx->t_start + std::chrono::duration_cast<clk::duration>(
          std::chrono::duration<double>(ctx->duration + 5.0));
      for (long k = 0;; ++k) {
        const double due_s = offset + (double)k * period;
        if (due_s >= ctx->duration) break;
        const auto due = ctx->t_start + std::chrono::duration_cast<clk::duration>(
            std::chrono::duration<double>(due_s));
        const auto now = clk::now();
        if (now > hard_stop) break;
        if (now < due) std::this_thread::sleep_until(due);
        else if (now - due > std::chrono::milliseconds(1) && due_s >= ctx->warmup) ctx->late->fetch_add(1);
        CUDA_CHECK(cudaMemcpyAsync(shm_in, buf.data() + fi * IN_BYTES, IN_BYTES, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        do_infer_and_post(fi);
        const double turnaround = std::chrono::duration<double, std::milli>(clk::now() - due).count();
        // The first second carries connection and first-execution costs a camera
        // that has been running pays once, not per frame; keep it out of the tail.
        if (due_s >= ctx->warmup) {
          std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(turnaround);
        }
        fi = (fi + 1) % n_frames;
      }
    } else {
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      CUDA_CHECK(cudaMemcpyAsync(shm_in, buf.data() + fi * IN_BYTES, IN_BYTES, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      double lat = do_infer_and_post(fi);
      { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }
      fi = (fi + 1) % n_frames;
    }
    }  // capacity (file) mode
  } else {
    // ---- RTSP: NVDEC zero-copy -> kernel writes into shm_in ----
    AVBufferRef* hw_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
      std::cerr << "cannot create CUDA hw device ctx" << std::endl; return;
    }
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, ctx->url.c_str(), nullptr, nullptr) < 0) {
      std::cerr << "stream " << ctx->stream_id << ": open failed" << std::endl; return;
    }
    avformat_find_stream_info(fmt, nullptr);
    int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    const AVCodec* dec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext* cctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(cctx, fmt->streams[vs]->codecpar);
    cctx->hw_device_ctx = av_buffer_ref(hw_ctx);
    cctx->get_format = [](AVCodecContext*, const enum AVPixelFormat* fmts) -> enum AVPixelFormat {
      for (const enum AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++)
        if (*p == AV_PIX_FMT_CUDA) return AV_PIX_FMT_CUDA;
      return fmts[0];
    };
    if (avcodec_open2(cctx, dec, nullptr) < 0) {
      std::cerr << "stream " << ctx->stream_id << ": codec open failed" << std::endl; return;
    }
    int src_w = cctx->width, src_h = cctx->height;
    float scale = std::min((float)IMG / src_w, (float)IMG / src_h);
    int nw = (int)(src_w * scale), nh = (int)(src_h * scale);
    int pad_x = (IMG - nw) / 2, pad_y = (IMG - nh) / 2;

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    dim3 blk(32, 16);
    dim3 grd((IMG + blk.x - 1) / blk.x, (IMG + blk.y - 1) / blk.y);

    auto t0 = std::chrono::steady_clock::now();
    while (true) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el > ctx->duration) break;
      if (av_read_frame(fmt, pkt) < 0) { av_seek_frame(fmt, -1, 0, AVSEEK_FLAG_BACKWARD); continue; }
      if (pkt->stream_index != vs) { av_packet_unref(pkt); continue; }
      if (avcodec_send_packet(cctx, pkt) < 0) { av_packet_unref(pkt); continue; }
      av_packet_unref(pkt);
      if (avcodec_receive_frame(cctx, frame) < 0) continue;
      if (frame->format != AV_PIX_FMT_CUDA) { av_frame_unref(frame); continue; }

      unsigned char* src_y = (unsigned char*)frame->data[0];
      unsigned char* src_uv = (unsigned char*)frame->data[1];
      int y_pitch = frame->linesize[0], uv_pitch = frame->linesize[1];
      if (!src_y || !src_uv) { av_frame_unref(frame); continue; }

      nv12_letterbox_kernel<<<grd, blk, 0, stream>>>(
          src_y, src_uv, y_pitch, uv_pitch, src_w, src_h, shm_in, nw, nh, pad_x, pad_y, IMG);
      CUDA_CHECK(cudaStreamSynchronize(stream));  // server reads shm_in only after kernel done
      long fi = 0;
      double lat = do_infer_and_post(fi);
      { std::lock_guard<std::mutex> lk(*ctx->lat_mtx); ctx->latencies->push_back(lat); }
      av_frame_unref(frame);
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&cctx);
    avformat_close_input(&fmt);
    if (hw_ctx) av_buffer_unref(&hw_ctx);
  }

  // shm_regions unregisters both regions as this function returns
  cudaStreamDestroy(stream);
  cudaFreeHost(h_dets); cudaFreeHost(h_count);
  cudaFree(d_dets); cudaFree(d_count);
}

int main(int argc, char** argv) {
  std::string url = "rtsp://localhost:8554/cam1", model = "yolov8s";
  std::string mode = "rtsp", file_path = "frames.bin";
  int streams = 1;
  double duration = 15.0;
  double cam_fps = 0;
  std::string phase = "random";
  unsigned seed = 1;
  double warmup = 1.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--url" && i + 1 < argc) url = argv[++i];
    else if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--file" && i + 1 < argc) file_path = argv[++i];
    else if (a == "--streams" && i + 1 < argc) streams = std::stoi(argv[++i]);
    else if (a == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
    else if (a == "--fps" && i + 1 < argc) cam_fps = std::stod(argv[++i]);
    else if (a == "--phase" && i + 1 < argc) phase = argv[++i];
    else if (a == "--seed" && i + 1 < argc) seed = (unsigned)std::stoul(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::stod(argv[++i]);
  }
  if (mode == "paced" && cam_fps <= 0) { std::cerr << "--mode paced needs --fps > 0" << std::endl; return 2; }
  if (phase != "random" && phase != "sync") { std::cerr << "--phase must be random|sync" << std::endl; return 2; }
  // One shared start line, far enough ahead that every camera has registered its
  // shm regions before its first frame is due.
  const auto t_start = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(2.0 + 0.05 * streams));
  std::atomic<long> late{0};
  if (mode == "file" || mode == "paced") {
    std::ifstream f(file_path, std::ios::binary);
    if (!f.good()) { std::cerr << "cannot open " << file_path << std::endl; return 2; }
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    g_frame_buf.resize(sz);
    f.read(g_frame_buf.data(), sz);
  }

  std::atomic<long> frames{0}, dets{0};
  std::vector<double> latencies; std::mutex lat_mtx;
  std::vector<std::thread> threads;
  std::vector<StreamCtx> ctxs(streams);
  for (int i = 0; i < streams; ++i) {
    ctxs[i] = {url, model, mode, file_path, i, streams, &frames, &dets, &latencies, &lat_mtx, duration};
    ctxs[i].fps = cam_fps; ctxs[i].phase = phase; ctxs[i].seed = seed; ctxs[i].warmup = warmup;
    ctxs[i].t_start = t_start; ctxs[i].late = &late;
    threads.emplace_back(run_stream, &ctxs[i]);
  }
  for (auto& t : threads) t.join();
  // Paced mode: an overloaded run keeps draining its backlog past the window, so
  // divide by the time actually taken, or delivered throughput is over-reported.
  double elapsed = duration;
  if (mode == "paced")
    elapsed = std::max(duration, std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count());

  long n = frames.load();
  std::sort(latencies.begin(), latencies.end());
  auto pct = [&](double p) -> double {
    if (latencies.empty()) return 0;
    return latencies[std::min((size_t)(p * latencies.size()), latencies.size() - 1)];
  };
  double sn = g_b2_stages.n ? g_b2_stages.n : 1;
  double lat_mean = 0;
  for (double v : latencies) lat_mean += v;
  if (!latencies.empty()) lat_mean /= latencies.size();
  const double lat_max = latencies.empty() ? 0 : latencies.back();
  printf("{\"pipeline\":\"cpp_grpc_cuda_shm\",\"mode\":\"%s\",\"model\":\"%s\",\"streams\":%d,"
         "\"frames\":%ld,\"detections\":%ld,\"fps\":%.2f,"
         "\"lat_ms_p50\":%.3f,\"lat_ms_p95\":%.3f,\"lat_ms_p99\":%.3f,"
         "\"lat_ms_max\":%.3f,\"lat_ms_mean\":%.3f,"
         "\"fps_per_camera\":%.2f,\"offered_fps\":%.1f,\"phase\":\"%s\",\"late_frames\":%ld,"
         "\"stages_ms\":{\"grpc_infer\":%.3f,\"post_compact\":%.3f,\"nms_cpu\":%.3f}}\n",
         mode.c_str(), model.c_str(), streams, n, dets.load(), n / elapsed,
         pct(0.50), pct(0.95), pct(0.99),
         lat_max, lat_mean, cam_fps, cam_fps * streams,
         mode == "paced" ? phase.c_str() : "", late.load(),
         g_b2_stages.grpc / sn, g_b2_stages.post / sn, g_b2_stages.nms / sn);
  return 0;
}
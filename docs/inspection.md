# Manufacturing inspection — the serving problem behind a visual QA line

[← index](../README.md) · prev: [Contention](contention.md) · next: [Inspection encoders](inspection-models.md)

Everything else in this study serves one detector on camera frames. A visual
inspection line is a different shape of problem: every part is photographed,
most parts are good, and the rare bad one has to be caught without anyone
having labelled what "bad" looks like. This section asks what that costs to
**serve**, on the same RTX 3090 and with the same measurement bar as the rest
of the repo.

**Inference only.** No training, no fine-tuning, no accuracy numbers. GPU time
depends on a network's structure, not on what it learned, so every model here
is timed with random weights. Anything that *would* depend on accuracy, such as
how often a part gets flagged, becomes a setting that is varied instead.

---

## Why there is no defect-trained detector

Defects are rare (often under 1% of parts) and new kinds keep appearing. A
supervised defect detector suffers both: too few examples of each class, and a
model that is out of date the day a new failure mode shows up. So the pipeline
needs **no defect labels at all**:

| Stage | Job | Runs on | Trained on |
|---|---|---|---|
| 1. Locate | yolov8s finds and crops each part (or a fixed crop, if camera and part are fixtured) | every frame | parts, which are in every image, so no imbalance |
| 2. Judge | an anomaly model scores each crop | every crop | good parts only |
| 3. Explain | an open-vocabulary model names or outlines what was flagged | a fraction *p* of flagged crops | nothing |

Stage 1 is already measured: yolov8s costs 0.99 ms of engine time at 640
([across architectures](model-zoo.md)) and 1.25 ms inside the A2 pipeline
([results](results.md)). A part locator does not need higher resolution, since
parts are large in the frame. **Resolution matters for stage 2**, which is
looking for scratches, so that is where the input size is varied.

A supervised defect detector can come later, once stage 2 has flagged enough
real defects to label. It would cost the same as the yolov8s already measured.

## What is measured, and where

| Phase | Question | Status |
|---|---|---|
| 1-pre | Which stage-2 encoders are fast enough to carry forward? | **Measured:** [inspection encoders](inspection-models.md) |
| 0 | Do big models (SAM, DINOv2-L, RT-DETR) gain from batching the way yolov8s does? | Planned |
| 1a | PatchCore: how much is the backbone and how much the nearest-neighbour search? | Planned |
| 1b | Dense anomaly maps: send the map back, or reduce it to a score in the graph? | Planned |
| 1c | Stage 3: does Grounding DINO convert to TensorRT, and what does it cost per phrase? | Planned |
| 2 | The whole pipeline under live camera load: do crops batch well, and does the rare heavy stage slow the fast path? | Planned |

Each phase is published on its own, with its predictions committed before the
run.

---

[← index](../README.md) · prev: [Contention](contention.md) · next: [Inspection encoders](inspection-models.md)

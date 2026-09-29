# Manufacturing inspection — now its own study

[← index](../README.md) · prev: [Contention](contention.md) · next: [Accuracy](accuracy.md)

This section grew into a study of its own, with its own repository and site:

**[Manufacturing inspection on one GPU →](https://sadbodhs.github.io/manufacturing_inspection/)** ·
[sadbodhs/manufacturing_inspection](https://github.com/sadbodhs/manufacturing_inspection)

It asks what a **label-free visual inspection line** costs to serve: a locator on every
frame, an anomaly model trained on good parts only on every part, and a rare, heavy
open-vocabulary model on the flagged ones. It uses this study's containers, engines and
measurement bar, and its live-line client is built on this study's paced B2/B3 client.

What it found, in one line each:

| Question | Finding |
|---|---|
| [Which stage-2 encoders are fast enough?](https://sadbodhs.github.io/manufacturing_inspection/encoders/) | All cost 0.29–1.35 ms at a 256 crop; EfficientAD-S is the most expensive CNN (1.05 ms), DINOv2-B the most expensive overall |
| [PatchCore: backbone or memory-bank search?](https://sadbodhs.github.io/manufacturing_inspection/patchcore-search/) | Below ~8k patches, searching inside the engine is cheaper than not searching |
| [Do FAISS or cuVS search the bank faster?](https://sadbodhs.github.io/manufacturing_inspection/embedding-search/) | No: TensorRT's brute force beats every index with recall ≥ 0.9 by 4× or more |
| [Do big models gain from batching?](https://sadbodhs.github.io/manufacturing_inspection/big-models/) | By architecture, not size: RT-DETR −37.5%, SAM-B +14.3%; random weights time like pretrained |
| [The whole line, under live load](https://sadbodhs.github.io/manufacturing_inspection/the-line/) | ~10 cameras per 3090 at 4 parts per frame; the rare heavy stage sets the tail and can collapse the line |

The encoder check was first run and published here (commits `4400d76`, `03ed4bf`); its
script and data remain in this repository, and continue in the new one.

---

[← index](../README.md) · prev: [Contention](contention.md) · next: [Accuracy](accuracy.md)

# Real-Time Motorcycle Helmet Detection Using YOLO11n and OpenCV

> **Project status: early planning.** Nothing described below is implemented yet.
> Every feature in this document is **planned**, not available.

## Problem statement

Riding a motorcycle without a helmet greatly increases the risk of serious head
injury. Manually checking video for helmet use is slow and error-prone. This
project explores whether a lightweight object detector can classify motorcycle riders' head regions as `with_helmet` or `without_helmet` in video captured from a webcam.

## Objectives

- Detect rider head regions classified as `with_helmet`. *(planned)*
- Detect rider head regions classified as `without_helmet`. *(planned)*
- Process live webcam video using OpenCV. *(planned)*
- Display bounding boxes, class labels and confidence scores. *(planned)*
- Display real-time detection statistics and FPS. *(planned)*

## Planned features

| Feature | Status |
|---|---|
| Live webcam capture with OpenCV | Planned |
| YOLO11n inference on CPU | Planned |
| Bounding boxes, labels and confidence scores | Planned |
| Per-class detection counts | Planned |
| FPS display | Planned |
| Configuration file (model path, camera index, thresholds) | Planned |
| Dataset evaluation and selection | Planned |
| Model training notebook for Google Colab | Planned |

## Planned technology stack

- **Language:** Python 3.10
- **Detection model:** YOLO11n (Ultralytics)
- **Video and drawing:** OpenCV
- **Training environment:** Google Colab
- **Tooling:** Git, GitHub, VS Code, Claude Code

## Hardware and approach

- **Training:** done on Google Colab, using a GPU runtime if available. Training
  is not run on the local machine.
- **Inference:** done locally on a Windows 11 laptop, using the **CPU only** and
  a built-in webcam.
- Frame rate on a CPU-only laptop is expected to be modest. It has not been
  measured yet. Options such as a smaller input size may be evaluated later.

## Proposed classes

Detections are intended to describe a **rider's head region**.

| Class | Intended meaning |
|---|---|
| `with_helmet` | The head region of a motorcycle rider who is wearing a helmet. |
| `without_helmet` | The head region of a motorcycle rider who is not wearing a helmet. |

The dataset has **not been selected**. We will not assume that any dataset has
matching annotations until it has been inspected. If a dataset uses different
class definitions, the labels may need to be reviewed or remapped.

## Planned workflow

1. Set up the repository, documentation and Python environment.
2. Research and evaluate candidate datasets, and inspect their annotations.
3. Prepare the chosen dataset and map its labels to the two classes.
4. Train YOLO11n on Google Colab and record the results.
5. Evaluate the trained model on held-out data.
6. Build the local webcam inference application with OpenCV.
7. Measure CPU performance and tune settings such as input size and threshold.
8. Document results, limitations and usage.

## Evaluation metrics

Planned metrics, to be reported once a model has been trained:

- **Precision** and **recall**, per class
- **mAP@0.5** and **mAP@0.5:0.95**
- **Confusion matrix**
- **Inference speed** (FPS and latency per frame) on the target CPU

## Limitations

- No model, dataset or application exists yet.
- Accuracy will depend on the dataset and may be lower in poor lighting, at
  unusual camera angles, at long distances, or with partial occlusion.
- A model trained on one dataset may not generalise to other regions, helmet
  styles or camera setups.
- Real-time speed on a CPU-only laptop is not guaranteed.
- A webcam in a lab setting is not representative of real road conditions.

## Educational-use disclaimer

This project is for **educational and research purposes only**. It is not a
certified safety system and must not be used for law enforcement, enforcement of
traffic rules, or any decision that affects a person's safety, rights or
liability. Detection results may be wrong. Anyone recording or analysing video of
people should follow applicable privacy laws and obtain any required consent.

## License

No license has been chosen yet.

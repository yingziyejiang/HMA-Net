# HMA-Net — final model code and parameters

Companion release for the manuscript *Incremental information auditing reveals non-incremental
anatomical inputs in dental age and sex estimation*.

## Contents

    train.py                  training entry point (two-phase fine-tuning)
    hma_net/hma_net.py        model, losses, dataset and evaluation (self-contained)
    config/parameters.json    parameters of the reported configuration
    figures/                  scripts for the Grad-CAM panels
    data_sample/              three de-identified sample cases and the label format
    weights/                  tooth-detector weights (see weights/LICENSE)

## Install

    pip install -r requirements.txt

## Model interface

    import torch, sys
    sys.path.insert(0, "hma_net")
    from hma_net import HMAAgePredictor

    model = HMAAgePredictor().eval()

The model expects one batch dict: img (B,3,256,256), teeth (B,32,3,64,64), pos_bbox (B,32,4),
fdi (B,32), mask (B,32). It returns (age_pred, gender_logit, coral_logits); the sex probability
is sigmoid(gender_logit).

## Train

    python train.py --manifest manifest.csv --outdir runs/hma_net

The manifest CSV needs the columns img_path, label_path, age, gender, split, with per-tooth label
files in the format described in data_sample/README.md. Protocol of the reported run: Adam, 15
frozen-backbone epochs at lr 1e-3 then full fine-tuning at lr 1e-4 with the learning rate halved
every 20 epochs, batch size 32 with a gender-balanced sampler, 100 epochs, best validation score
per head.

## Data

The clinical cohort is not distributed here; the manuscript states how it can be requested. The
cases in data_sample/ are de-identified and included for illustration and smoke-testing only.

## Detector

weights/tooth_detector_yolov8m_obb.pt — YOLOv8m-OBB, 52 FDI classes. Inference settings used in
the manuscript: confidence 0.25, NMS IoU 0.7, max_det 300, imgsz 1024. Licence: AGPL-3.0 (see
weights/LICENSE).

    from ultralytics import YOLO
    m = YOLO("weights/tooth_detector_yolov8m_obb.pt")
    res = m(image_path, conf=0.25, iou=0.7, max_det=300, imgsz=1024)[0]

## Grad-CAM

    python figures/gradcam_l1_panoramic.py --checkpoint <ckpt> --image <opg.jpg> --rois <labels> --stream both
    python figures/gradcam_l3_per_tooth_globalnorm.py --checkpoint <ckpt> --image <opg.jpg> --rois <labels> --fdi 46,36 --stream l3 --target sex

## Citation

See CITATION.cff.

## Licence

Code: MIT (LICENSE). Detector weights: AGPL-3.0. Research use only - not for clinical diagnosis,
treatment decisions, age assessment of minors, or forensic or immigration procedures.

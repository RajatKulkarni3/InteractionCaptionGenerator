# Human Action Recognition Pipeline

Video -> Pose Estimation -> Object Detection -> Human-Object Relation -> Action Prediction -> Explanation Generation

## What's in this zip, honestly

| File | Status |
|---|---|
| `pose_estimation_final.py` | Working. Posture classification (standing/sitting/crouching/lying/running/jumping) from YOLO11-Pose keypoints, with a thigh-angle fallback tier for when ankles aren't visible. |
| `interaction_engine.py` | Working. Rule-based, explainable interaction detection (Action / Reason list / Confidence %) using object tracking + timed keypoint-to-object proximity. Currently implements `drinking` and `using phone` as worked examples. |
| `main.py` | Working, but **still wired to the old BLIP captioner (`HOICaptioner`)**, not yet switched over to `interaction_engine.py`. That rewire is a short follow-up, not done in this drop. |
| `training/feature_extraction.py` | Working code, **not run against real data yet** -- no dataset was downloaded in the environment this was built in (network-restricted sandbox, no dataset host access). |
| `training/train.py` | Working code, verified end-to-end with a synthetic smoke-test dataset (60/40 split, training, evaluation, model saving all confirmed to run correctly). **Not yet trained on real images** -- there is no `model.joblib` in this zip because no real training has happened yet. |

**There is no pretrained model file in this zip.** Handing you one right now would mean either fabricating results or training on fake data and calling it real -- neither is something I'll do. What's here is the complete, working toolchain to produce a genuinely trained-and-evaluated model yourself, in minutes once you have the dataset.

## Step 1: Get a dataset

**Quick start (recommended first):** the HAR dataset -- 15 action classes as labeled images, including `drinking`, `calling`, `texting`, `sitting`, `eating`, `using_laptop`. ~330MB, no application needed.
- Kaggle: https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset
- Or via Hugging Face: `pip install datasets` then `load_dataset("Bingsu/Human_Action_Recognition")`

Arrange it (or export it) into this layout:
```
dataset_root/
    drinking/
        img001.jpg
        ...
    calling/
        ...
    sitting/
        ...
```

**Important limit:** this is single images, not video. It can validate the instantaneous geometry (hand-near-object, face-near-object, posture angles) but NOT the duration-based rules in `interaction_engine.py` (e.g. "near mouth for 1.8 seconds") -- there's no time axis in a photo.

**For real video + duration validation later:** NTU RGB+D 60/120 (`rose1.ntu.edu.sg/dataset/actionRecognition/`) has daily actions as actual video with 3D skeletons, including `drink water`, `eat meal/snack`, `sitting down`, `standing up`. Free for academic use, but requires a short registration/release-agreement step, and its skeleton format (25 Kinect joints) needs remapping to your COCO-17 format.

## Step 2: Extract features

```bash
cd training
python feature_extraction.py --dataset_root /path/to/dataset_root --out features.csv
```

This runs your existing pose + object detection models over every image and writes one row per detected person: image path, label, and the same geometric features `interaction_engine.py` uses live (wrist-to-object distance, face-to-object distance, knee/thigh/torso angles), normalized by bbox height.

## Step 3: Train and evaluate

```bash
python train.py --features features.csv
```

This does a **stratified 60/40 train/test split** (60% train, 40% held out), trains a RandomForest on the 60%, and evaluates ONLY on the 40% it never saw. It prints and saves:
- overall test-set accuracy
- a per-class precision/recall/F1 report (so you see which specific actions are weak, not one blended number)
- a confusion matrix (what's getting confused with what)
- feature importances (which signals are actually doing the work)

Outputs: `model.joblib` (the trained model) and `evaluation_report.txt` (the full numbers, for your records / report).

## Step 4: Plug the trained model into live inference

Not yet wired up in this drop. Once you have `model.joblib`, the next step is loading it in `main.py` and calling `pipeline.predict_proba(features)` on the same feature vector computed live each frame -- happy to build that next.

## Still pending / next steps

- Rewire `main.py` to use `interaction_engine.py` instead of `HOICaptioner`.
- Add a `sitting_on_chair`-style action to `interaction_engine.py` (hip/knee proximity rather than hand/mouth) as a template for posture-based interactions.
- Load `model.joblib` into live inference once it's trained on real data.
- If you want the duration logic properly validated, work through NTU RGB+D access.

# Vision Demand Scales to Estimate Object Detection Capabilities

# Current Implementation
## Methodology 

Based on OECD vision capability scale:
1. Dynamic versus static input
2. Sensor type and resolution
3. Mobile versus static camera
4. Real versus staged or synthetic environment
5. Visual task
6. Environmental complexity
7. Environment type
8. Lighting and weather
9. Real-time requirements
10. Visual-only versus multimodal input

The work combines it into a single image-demand scale. Where it is rephrased to focus con the capability required for the instance, focusing on image quality, where aplications are removed. 

Each experiment configuration selects its zero-shot and/or few-shot prompt files. Version
16 supports both strategies and uses `data/prompts/16/images` for the few-shot examples.
Version 17 currently defines the new observable-demand rubric as a zero-shot prompt.

Same rubric used for two tasks:
- Localisation, the prompt defines the task as determining object positions using tight rectangular bounding boxes.
- Object detection, it defines the task as localisation plus assigning each box to one of the available categories.

Pipeline:
1. Insert the task definition into the modified rubric.
2. Provide either the rubric alone or the rubric plus few-shot examples.
3. Submit the image to the configured vision-language model.
4. Parse the output into an integer from 1 to 5.
5. Group images according to their predicted demand level.
6. Measure the performance of several object detectors inside each group.
7. Plot detector performance against demand level as a model characteristic curve.

Combining datasets is intentional: the pooled population is meant to cover a broad range of demands using easy and difficult images from different datasets. Demand level may nevertheless correlate with dataset or domain, so a pooled curve can reflect both increasing visual demand and changes in dataset composition across levels. Because the included image domains are related, this is expected to be a gradual composition shift rather than a complete domain change, but it remains an interpretation caveat. Dataset-specific analyses can be reported as diagnostics alongside the primary pooled curves.

## Dataset partitions

The pipeline uses three global, version-independent partitions:

- `calibration`: 500 images from each dataset's training split
- `mllm_selection`: 2,000 different images from each training split
- `locked_confirmation`: the complete validation split

The definitions and dataset locations are stored in `configs/splits.toml`. Experiment
version 16 is configured in `configs/experiments/v16.toml` and later versions should
refer to the same global split configuration. Stable manifests are stored in
`data/splits/<partition>.csv`, not below a rubric-version directory.

Generate all three manifests once:

```bash
poetry run python scripts/preparation/get_image_ids.py --partition all
```

The command is deliberately non-destructive. Pass `--overwrite` only when intentionally
redefining the global experimental population. The generator samples deterministically,
keeps Roboflow variants of the same driving source image together, and rejects overlap
between partitions. It loads IDs from the datasets themselves rather than from one
detector's prediction file.

COCO 2017 and VOC 2007 are loaded through FiftyOne. The downloaded Roboflow/Udacity
driving export is flat and contains two filename variants for each of 15,000 source
frames. Prepare one canonical image per source and a capture-sequence-disjoint split:

```bash
poetry run python scripts/preparation/prepare_driving_dataset.py
```

This creates `../vision_datasets/self-driving-car-v2-split/train` with 2,509 images from
six complete capture sequences and `validation` with the remaining 12,491 images. Each
directory contains its own `_annotations.coco.json`; `configs/splits.toml` points to these
paths. The original archive, extracted flat export, and previous `driving-validation`
directory are preserved.

## Object-detection pipeline commands

All commands require an experiment version and one of `calibration`, `mllm_selection`,
or `locked_confirmation`; no partition is selected implicitly. Prediction-confidence
filtering uses each model's native default, except for the Hugging Face-backed D-FINE,
RT-DETR-v2, and DETR models, whose FiftyOne wrapper otherwise retains every decoder
query. These models use a confidence threshold of 0.5 by default, configurable with
`--transformer-confidence-threshold`. Detection and localization matching use a common
IoU threshold of 0.5; this is a matching threshold, not a prediction-confidence threshold.

Before running each model, the inference script clears the shared FiftyOne `predictions` field. Inference failures are fatal, and an output file is saved only after every sample has received a fresh prediction container. This prevents a failed model from inheriting the preceding model's predictions. A local compatibility shim also handles RF-DETR versions that expose COCO class names as a list while using sparse COCO category IDs.

The configured closed-set panel contains 40 models: five YOLOv5 sizes, five YOLOv8 sizes, two YOLOv9 sizes, five YOLOv10 sizes, five YOLO11 sizes, five D-FINE sizes, five RF-DETR sizes, five RT-DETR/RT-DETR-v2 variants, DETR, Faster R-CNN, and RetinaNet. Open-vocabulary detectors are not included in this panel.

Generate raw ground-truth and prediction files named `<model>_predictions.json` for every dataset and model configured in the script:

```bash
poetry run python scripts/inference/get_predictions.py --version 16 --partition calibration
```

By default, inference overwrites existing model/dataset result files. Resume an interrupted run without recomputing completed outputs with:

```bash
poetry run python scripts/inference/get_predictions.py --version 16 --partition calibration --skip-existing
```

Run or replace only selected models with `--models`. Do not combine this with `--skip-existing` when replacing invalid outputs. For example, rerun the five RF-DETR models over all three datasets with:

```bash
poetry run python scripts/inference/get_predictions.py --version 16 --partition calibration --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
```

Generate both the class-aware detection and class-agnostic localization evaluations for each dataset. Each output is named `<model>_detection_and_localization_evaluation.json`:

```bash
poetry run python scripts/evaluation/detection.py coco-2017 --version 16 --partition calibration
poetry run python scripts/evaluation/detection.py voc-2007 --version 16 --partition calibration
poetry run python scripts/evaluation/detection.py driving --version 16 --partition calibration
```

By default, these commands overwrite existing combined evaluations. Resume an interrupted evaluation run with:

```bash
poetry run python scripts/evaluation/detection.py coco-2017 --version 16 --partition calibration --skip-existing
poetry run python scripts/evaluation/detection.py voc-2007 --version 16 --partition calibration --skip-existing
poetry run python scripts/evaluation/detection.py driving --version 16 --partition calibration --skip-existing
```

Evaluation can likewise be restricted to selected models. After replacing the RF-DETR predictions, overwrite only their evaluations with:

```bash
poetry run python scripts/evaluation/detection.py coco-2017 --version 16 --partition calibration --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
poetry run python scripts/evaluation/detection.py voc-2007 --version 16 --partition calibration --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
poetry run python scripts/evaluation/detection.py driving --version 16 --partition calibration --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
```

Combine the evaluations into the per-image metrics table:

```bash
poetry run python scripts/evaluation/aggregate_evaluation_results.py --version 16 --partition calibration
```

Outputs are written below `outputs/object_detection/v<version>/<partition>/`.

The repository-root `run_partition_pipeline.sh` runs inference, evaluation, MLLM
annotation, aggregation, and all overall and per-family plots for every dataset. Edit
`VERSION`, `PARTITION`, `PROMPT_STRATEGY`, and `OVERWRITE` at the top before running it. The local
`llama-server` must already be running when the annotation stage begins.


## Local Hugging Face annotation

The annotation script uses a local OpenAI-compatible `llama-server`.

Hugging Face model installation:

```bash
poetry run hf download DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF \
  Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q6_K.gguf \
  mmproj-F16.gguf \
  --local-dir ../data/models/qwen3.6-27b-q6
```

Start the local vision server in a separate terminal:

```bash
llama-server \
  -m ../data/models/qwen3.6-27b-q6/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q6_K.gguf \
  --mmproj ../data/models/qwen3.6-27b-q6/mmproj-F16.gguf \
  -ngl 999 \
  -c 16384 \
  --host 127.0.0.1 \
  --port 8080
```

The annotation script reads image IDs and paths from the selected global manifest. Its
fourth positional argument selects `zeroshot` or `fewshot`. It annotates the complete
partition by default; use `--max-samples N` only for an explicit partial run. Existing
valid rows are resumed unless `--overwrite` is passed. Strategy-specific filenames keep
zero-shot and few-shot annotations separate.

```bash
poetry run python scripts/rubrics/llm-prompting.py coco-2017 detection 17 zeroshot --partition calibration
poetry run python scripts/rubrics/llm-prompting.py voc-2007 detection 17 zeroshot --partition calibration
poetry run python scripts/rubrics/llm-prompting.py driving detection 17 zeroshot --partition calibration

poetry run python scripts/rubrics/llm-prompting.py coco-2017 localization 17 zeroshot --partition calibration
poetry run python scripts/rubrics/llm-prompting.py voc-2007 localization 17 zeroshot --partition calibration
poetry run python scripts/rubrics/llm-prompting.py driving localization 17 zeroshot --partition calibration
```
## Analysis and plots

Merge all datasets:

The merge retains every available annotation and records the source dataset, so later analysis identifies images by both dataset and image ID.

```bash
poetry run python scripts/analysis/merged_dataset_distribution.py 17 detection zeroshot --partition calibration
poetry run python scripts/analysis/merged_dataset_distribution.py 17 localization zeroshot --partition calibration
```

Generate overall curves:
```bash
poetry run python scripts/analysis/model_vs_gpt_difficulty.py 17 detection zeroshot --partition calibration
poetry run python scripts/analysis/model_vs_gpt_difficulty.py 17 localization zeroshot --partition calibration
```

Generate curves for every model family. Each family figure contains a separate
curve for every available model size in that family:
```bash
poetry run python scripts/analysis/permodel_vs_gpt_difficulty.py 17 detection zeroshot --partition calibration
poetry run python scripts/analysis/permodel_vs_gpt_difficulty.py 17 localization zeroshot --partition calibration
```

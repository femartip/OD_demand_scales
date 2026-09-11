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

Each image is annotated once using the object-detection task definition (tight
bounding boxes plus category assignment). The same demand level is evaluated
separately against localisation and class-aware detection performance.

Pipeline:
1. Insert the object-detection task definition into the modified rubric.
2. Provide either the rubric alone or the rubric plus few-shot examples.
3. Submit the image to the configured vision-language model.
4. Parse the output into an integer from 1 to 5.
5. Group images according to their predicted demand level.
6. Measure the performance of several object detectors inside each group.
7. Plot detector performance against demand level as a model characteristic curve.

The active dataset is **COCO-ReM**: COCO 2017 images and the same 80 categories,
with refined instance annotations. Detection boxes and areas are computed from the
ReM masks, rather than trusting the supplied boxes. Zero-area masks are omitted;
all images, category IDs, annotation IDs, and crowd flags are retained. Original
masks remain available in the downloaded release archives.

## Dataset partitions

The active ReM sampling protocol uses seed 16 and no detector outcomes:

- `calibration`: 1,000 random training images plus 500 selected by observable properties
- `mllm_selection`: 6,000 random training images reserved first, excluding prior calibration IDs
- `locked_confirmation`: all 5,000 validation images

`configs/splits.toml` defines these global, version-independent partitions. Their
COCO-only manifests are in `data/splits/coco-rem/<partition>.csv`. The previous
mixed-dataset manifests and results are retained as historical artifacts.

Prepare the ReM annotations before running inference:

```bash
poetry install
poetry run python scripts/preparation/prepare_coco_rem.py
```

This downloads the official train and validation releases into
`outputs/datasets/coco-rem/`, derives detection annotations, and records source URLs,
SHA-256 hashes, corrected-box counts, and omitted annotation IDs. Existing release
ZIPs are reused. Use `--split train` or `--split validation` to prepare one split.

The supplementary calibration sample contains 100 images each for small typical
object size, many objects, high box overlap, many categories, and crowd presence.
The first four use fixed 10% tail rules. Selection is random within these groups,
without duplicate images. Manifests retain
`selection_group` and observable feature columns; `sampling_summary.json` records
thresholds and annotation hashes.

The generator samples the full ReM annotation inventory, then downloads only the
selected images. Generate all three partitions together:

```bash
poetry run python scripts/preparation/get_image_ids.py --partition all
```

The generator refuses existing manifests unless `--overwrite` is supplied. It uses
the existing seed and partition sizes and rejects overlapping image IDs. Inference
loads the manifest image paths with the prepared ReM labels in `ground_truth`.
VOC and Driving are no longer part of the active pipeline.

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

Generate raw ground-truth and prediction files named `<model>_predictions.json` for COCO-ReM and every configured model:

```bash
poetry run python scripts/inference/get_predictions.py --version 16 --partition calibration
```

By default, inference overwrites existing model/dataset result files. Resume an interrupted run without recomputing completed outputs with:

```bash
poetry run python scripts/inference/get_predictions.py --version 16 --partition calibration --skip-existing
```

Run or replace only selected models with `--models`. Do not combine this with `--skip-existing` when replacing invalid outputs. For example, rerun the five RF-DETR models over COCO-ReM with:

```bash
poetry run python scripts/inference/get_predictions.py --version 16 --partition calibration --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
```

Generate both the class-aware detection and class-agnostic localization evaluations for each dataset. Each output is named `<model>_detection_and_localization_evaluation.json`:

```bash
poetry run python scripts/evaluation/detection.py coco-rem --version 16 --partition calibration
```

By default, these commands overwrite existing combined evaluations. Resume an interrupted evaluation run with:

```bash
poetry run python scripts/evaluation/detection.py coco-rem --version 16 --partition calibration --skip-existing
```

Evaluation can likewise be restricted to selected models. After replacing the RF-DETR predictions, overwrite only their evaluations with:

```bash
poetry run python scripts/evaluation/detection.py coco-rem --version 16 --partition calibration --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
```

Combine the evaluations into the per-image metrics table:

```bash
poetry run python scripts/evaluation/aggregate_evaluation_results.py --version 16 --partition calibration
```

Detector predictions and evaluations are shared by all rubric versions and are written
below `outputs/object_detection/coco-rem/<partition>/`. Rubric annotations and figures remain
versioned under `outputs/annotations/coco-rem/v<version>/<partition>/` and
`outputs/figures/coco-rem/v<version>/<partition>/`. These paths prevent reuse of
metrics computed with the original annotations.

The repository-root `run_partition_pipeline.sh` runs inference, evaluation, MLLM
annotation, aggregation, and all overall and per-family plots for COCO-ReM. Edit
`VERSION`, `PARTITION`, `PROMPT_STRATEGY`, and `OVERWRITE` at the top before running it. The local
`llama-server` must already be running when the annotation stage begins.


## Local Hugging Face annotation

The annotation script uses a local OpenAI-compatible `llama-server`.

Hugging Face model installation:

```bash
poetry run hf download bartowski/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-Q5_K_M.gguf \
  mmproj-Qwen3.8-27B-f16.gguf \
  --local-dir ../data/models/qwen3.8-27b-q5
```

Start the local vision server in a separate terminal:

```bash
llama-server \
  -m ../data/models/qwen3.8-27b-q5/Qwen3.8-27B-Q5_K_M.gguf \
  --mmproj ../data/models/qwen3.8-27b-q5/mmproj-Qwen3.8-27B-f16.gguf \
  -ngl 999 \
  -np 4 \
  -c 65536 \
  -fa on \
  --image-min-tokens 1024 \
  --reasoning-budget 10000 \
  --reasoning-budget-message "I have gathered enough evidence. I will now state the single overall level." \
  --host 127.0.0.1 \
  --port 8080
```

Reasoning is enabled for every assessment. 

Annotate a partition with:

```bash
poetry run python scripts/rubrics/llm-prompting.py coco-rem 17 zeroshot --partition calibration --workers 4
```

The annotation script reads image IDs and paths from the selected global manifest. Its
third positional argument selects `zeroshot` or `fewshot`. It annotates the complete
partition by default; use `--max-samples N` only for an explicit partial run. Existing
valid rows are resumed unless `--overwrite` is passed. Strategy-specific filenames keep
zero-shot and few-shot annotations separate. Shared annotations retain the
`detection` filename prefix so existing detection labels can be resumed. No second
localisation annotation pass is needed.

```bash
poetry run python scripts/rubrics/llm-prompting.py coco-rem 17 zeroshot --partition calibration

```
## Analysis and plots

Prepare the shared COCO-ReM annotations for each task analysis:

Both merge commands read the shared `detection` annotation files and write the
task-specific tables expected by the plotting scripts. Historical localisation
annotation files are no longer used by this workflow. The merge records the source
dataset, so later analysis identifies images by both dataset and image ID. Rerun
both merge commands before plotting existing results with shared annotations.

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

Compare rubric versions on predictive accuracy and failure detection:

```bash
poetry run python scripts/analysis/evaluate_rubrics.py 16 17 18 detection zeroshot --partition calibration
```


## Baselines

Reference predictors for the rubric. Each writes `dataset,image_id,level` to
`outputs/baselines/<partition>/`, and `evaluate_rubrics.py` accepts their names alongside
rubric versions, so everything is scored with the same folds and metrics.

- `ionescu_features.py`: frozen ResNet-50 features and ridge regression trained on the
  detector score (Ionescu et al., CVPR 2016, retargeted). Supervised, so an upper
  reference rather than a peer.
- `ic9600_complexity.py`: IC9600 image complexity (Feng et al., TPAMI 2023), zero-shot.
  Needs a clone of https://github.com/tinglyfeng/IC9600 for `ICNet.py` and its checkpoint
  at `data/models/ic9600/ck.pth`.
- `mllm_direct.py`: the same annotator and task definition, without the rubric.
- `mllm_count.py`: the same annotator asked only for an object count, binned to five levels.

```bash
poetry run python scripts/baselines/ionescu_features.py --partition calibration
poetry run python scripts/baselines/ic9600_complexity.py --partition calibration
poetry run python scripts/baselines/mllm_direct.py --partition calibration --workers 4
poetry run python scripts/baselines/mllm_count.py --partition calibration --workers 4
```

The two MLLM baselines need the local `llama-server` and resume from their `_raw.csv`.
Counting produces much longer reasoning than rating, so it is several times slower; lower
the server `--reasoning-budget` to speed it up.

Score baselines and rubrics together:

```bash
poetry run python scripts/analysis/evaluate_rubrics.py 16 17 18 ionescu ic9600 mllm_direct mllm_count detection zeroshot --partition calibration
```

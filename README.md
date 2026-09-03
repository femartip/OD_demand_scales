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

The active few-shot prompt is stored in `data/prompts/few_shot_prompt_1dim_rubric.txt`, and `data/prompts/images` contains its example images. The one-shot prompt file is retained as a reference artifact.

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

## Object-detection pipeline commands

The inference script loads the COCO 2017 and VOC 2007 validation splits through FiftyOne. It expects the driving dataset in COCO format at `../vision_datasets/driving-validation/`. Prediction-confidence filtering uses each model's native default, except for the Hugging Face-backed D-FINE, RT-DETR-v2, and DETR models, whose FiftyOne wrapper otherwise retains every decoder query. These models use a confidence threshold of 0.5 by default, configurable with `--transformer-confidence-threshold`. Detection and localization matching use a common IoU threshold of 0.5; this is a matching threshold, not a prediction-confidence threshold.

Before running each model, the inference script clears the shared FiftyOne `predictions` field. Inference failures are fatal, and an output file is saved only after every sample has received a fresh prediction container. This prevents a failed model from inheriting the preceding model's predictions. A local compatibility shim also handles RF-DETR versions that expose COCO class names as a list while using sparse COCO category IDs.

The configured closed-set panel contains 40 models: five YOLOv5 sizes, five YOLOv8 sizes, two YOLOv9 sizes, five YOLOv10 sizes, five YOLO11 sizes, five D-FINE sizes, five RF-DETR sizes, five RT-DETR/RT-DETR-v2 variants, DETR, Faster R-CNN, and RetinaNet. Open-vocabulary detectors are not included in this panel.

Generate raw ground-truth and prediction files named `<model>_predictions.json` for every dataset and model configured in the script:

```bash
poetry run python scripts/inference/get_predictions.py
```

By default, inference overwrites existing model/dataset result files. Resume an interrupted run without recomputing completed outputs with:

```bash
poetry run python scripts/inference/get_predictions.py --skip-existing
```

Run or replace only selected models with `--models`. Do not combine this with `--skip-existing` when replacing invalid outputs. For example, rerun the five RF-DETR models over all three datasets with:

```bash
poetry run python scripts/inference/get_predictions.py --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
```

Generate both the class-aware detection and class-agnostic localization evaluations for each dataset. Each output is named `<model>_detection_and_localization_evaluation.json`:

```bash
poetry run python scripts/evaluation/detection.py coco-2017
poetry run python scripts/evaluation/detection.py voc-2007
poetry run python scripts/evaluation/detection.py driving
```

By default, these commands overwrite existing combined evaluations. Resume an interrupted evaluation run with:

```bash
poetry run python scripts/evaluation/detection.py coco-2017 --skip-existing
poetry run python scripts/evaluation/detection.py voc-2007 --skip-existing
poetry run python scripts/evaluation/detection.py driving --skip-existing
```

Evaluation can likewise be restricted to selected models. After replacing the RF-DETR predictions, overwrite only their evaluations with:

```bash
poetry run python scripts/evaluation/detection.py coco-2017 --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
poetry run python scripts/evaluation/detection.py voc-2007 --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
poetry run python scripts/evaluation/detection.py driving --models rfdetr-nano-coco-torch rfdetr-small-coco-torch rfdetr-medium-coco-torch rfdetr-base-coco-torch rfdetr-large-coco-torch
```

Combine the evaluations into the per-image metrics table:

```bash
poetry run python scripts/evaluation/aggregate_evaluation_results.py
```

Prepare the image-ID files used by the annotation stage:

```bash
poetry run python scripts/preparation/get_image_ids.py coco-2017
poetry run python scripts/preparation/get_image_ids.py voc-2007
poetry run python scripts/preparation/get_image_ids.py driving
```


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

Creating the image-ID files:

```bash
poetry run python scripts/preparation/get_image_ids.py coco-2017
poetry run python scripts/preparation/get_image_ids.py voc-2007
poetry run python scripts/preparation/get_image_ids.py driving
```


Run annotation:

```bash
poetry run python scripts/rubrics/llm-fewshot.py coco-2017 detection 16
poetry run python scripts/rubrics/llm-fewshot.py voc-2007 detection 16
poetry run python scripts/rubrics/llm-fewshot.py driving detection 16

poetry run python scripts/rubrics/llm-fewshot.py coco-2017 localization 16
poetry run python scripts/rubrics/llm-fewshot.py voc-2007 localization 16
poetry run python scripts/rubrics/llm-fewshot.py driving localization 16
```
## Analysis and plots

Merge all datasets:

The merge retains every available annotation and records the source dataset, so later analysis identifies images by both dataset and image ID.

```bash
poetry run python scripts/analysis/merged_dataset_distribution.py 16 detection
poetry run python scripts/analysis/merged_dataset_distribution.py 16 localization
```

Generate overall curves:
```bash
poetry run python scripts/analysis/model_vs_gpt_difficulty.py 16 detection
poetry run python scripts/analysis/model_vs_gpt_difficulty.py 16 localization
```

Generate curves for every model family. Each family figure contains a separate
curve for every available model size in that family:
```bash
poetry run python scripts/analysis/permodel_vs_gpt_difficulty.py 16 detection
poetry run python scripts/analysis/permodel_vs_gpt_difficulty.py 16 localization
```

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

Files data/prompts/few_shot_prompt_1dim_rubric.txt and data/prompts/one_shot_prompt_1dim_rubric.txt contain the prompts used, data/prompts/images contains the example images for the few shot prompt. 

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

## Object-detection pipeline commands

The inference script loads the COCO 2017 and VOC 2007 validation splits through FiftyOne. It expects the driving dataset in COCO format at `../vision_datasets/driving-validation/`.

Generate the class-aware object-detection evaluations for every dataset and model configured in the script:

```bash
poetry run python scripts/inference/get_predictions.py
```

Generate the class-agnostic localization evaluations for each dataset:

```bash
poetry run python scripts/evaluation/detection.py coco-2017
poetry run python scripts/evaluation/detection.py voc-2007
poetry run python scripts/evaluation/detection.py driving
```

Combine the evaluations into the per-image metrics table:

```bash
poetry run python scripts/analysis/get_detection_accuracy.py
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
```bash
poetry run python scripts/analysis/merged_dataset_distribution.py 16 detection
poetry run python scripts/analysis/merged_dataset_distribution.py 16 localization
```

Generate overall curves:
```bash
poetry run python scripts/analysis/model_vs_gpt_difficulty.py 16 detection
poetry run python scripts/analysis/model_vs_gpt_difficulty.py 16 localization
```

Per model family object curves:
```bash
poetry run python scripts/analysis/permodel_vs_gpt_difficulty.py 16 detection yolov8
poetry run python scripts/analysis/permodel_vs_gpt_difficulty.py 16 localization yolov8
```

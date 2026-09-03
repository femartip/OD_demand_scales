import os
import requests
import base64
import csv
import pandas as pd
import re
import sys
import argparse
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment import (PARTITIONS, annotations_dir, load_experiment_config, load_manifest, resolve_repo_path, split_config_for_version,)


def encode_image(image_path):
  return base64.b64encode(open(image_path, 'rb').read()).decode('ascii')


def build_prompt_content(prompt_text, image_dir):
  content = []
  for section in re.split(r"(\[IMAGE:[^\]]+\])", prompt_text):
    image_match = re.fullmatch(r"\[IMAGE:([^\]]+)\]", section)
    if image_match:
      encoded_image = encode_image(image_dir / image_match.group(1))
      content.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},})
    elif section:
      content.append({"type": "text", "text": section})
  return content

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=("coco-2017", "voc-2007", "driving"))
parser.add_argument("task", choices=("detection", "localization"))
parser.add_argument("version", type=int)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--max-samples", type=int, default=int(os.environ["MAX_SAMPLES"]) if "MAX_SAMPLES" in os.environ else None, help="Annotate only the first N manifest rows; by default annotate the whole partition",)
parser.add_argument("--max-label-attempts", type=int, default=int(os.environ.get("MAX_LABEL_ATTEMPTS", 5)), )
parser.add_argument("--overwrite", action="store_true", help="Replace this model's existing annotation file instead of resuming it", )
args = parser.parse_args()

dataset = args.dataset
max_samples = args.max_samples
max_label_attempts = args.max_label_attempts
task_to_evaluate = args.task

if task_to_evaluate == "localization":
   task = "localization"
   task_definition = "Localization consists of determining the position of the objects in a given image, i.e. generating a rectangular bounding box that tightly frames each detected object"

elif task_to_evaluate == "detection":
   task = "object detection"
   task_definition = "Object detection consists of determining the position of the objects in a given image, i.e. generating a rectangular bounding box that tightly frames each detected object and then establishing which of the available categories each one belongs to"

else:
   print("Not implemented")
   sys.exit(1)



version = args.version
experiment_config = load_experiment_config(version)
split_config = split_config_for_version(version)

MODEL_ID = "DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF"
MODEL_NAME = "qwen3.6-27b-q6"
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
image_ids = load_manifest(args.partition, dataset, split_config)
if max_samples is not None:
  if max_samples < 1:
    parser.error("--max-samples must be positive")
  image_ids = image_ids.iloc[:max_samples]

# check if those instances have been already labelled
output_annotations_dir = annotations_dir(version, args.partition)
os.makedirs(output_annotations_dir, exist_ok=True)
destination_path = output_annotations_dir / f"v{version}_{task_to_evaluate}_fewshot_labelled_images_{dataset}_{MODEL_NAME}.csv"
if args.overwrite and destination_path.is_file():
  destination_path.unlink()
labelled_prev = False
already_labelled = []
if os.path.isfile(destination_path):
  labelled_prev = True
  already_labelled_df = pd.read_csv(destination_path, delimiter=";", dtype={'image_id': object, 'level': object})

  for i,row in already_labelled_df.iterrows():
    if pd.notna(row["level"]) and str(row["level"]).strip().lower() != "nan":
      already_labelled.append(str(row["image_id"]))


few_shot_dir = resolve_repo_path(experiment_config["rubric"]["examples"])
prompt_path = resolve_repo_path(experiment_config["rubric"]["prompt"])
prompt_template = prompt_path.read_text(encoding="utf-8")
prompt_template = prompt_template.replace("[TASK_DEFINITION]", task_definition).replace("[TASK]", task)
system_prompt, user_prompt = prompt_template.split("[TARGET_IMAGE]", maxsplit=1)
system_content = build_prompt_content(system_prompt, few_shot_dir)


with open(destination_path, 'a', newline='', encoding='utf-8') as CSV_file:
    writer_CSV = csv.writer(CSV_file, delimiter=';')

    if not labelled_prev:
      writer_CSV.writerow(['image_id', 'level'])

    for _, row in image_ids.iterrows():
        image_id = str(row["image_id"])
        print(image_id)

        if image_id in already_labelled:
          print("Labelled!")
          continue

        IMAGE_PATH = Path(row["filepath"])
        if not IMAGE_PATH.is_file():
          raise FileNotFoundError(f"Manifest image not found: {IMAGE_PATH}")
        encoded_image = encode_image(IMAGE_PATH)
        headers = {"Content-Type": "application/json"}

        # Payload for the request
        payload = {"model": MODEL_ID, "messages": [{"role": "system","content": system_content,},{"role": "user","content": [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},{"type": "text","text": user_prompt,},]},],"temperature": 0,"top_p": 0.95,"chat_template_kwargs": {"enable_thinking": True}}
        final_text = ""
        for attempt in range(max_label_attempts):
          response = requests.post(MODEL_ENDPOINT, headers=headers, json=payload, timeout=600)
          response.raise_for_status()
          content = response.json()['choices'][0]["message"]["content"]
          final_text = content.strip() if content is not None else ""
          if final_text and final_text.lower() != "nan":
            break
          print(f"No level returned for {image_id}. Attempt {attempt + 1}/{max_label_attempts}.")

        if not final_text or final_text.lower() == "nan":
          final_text = None
        writer_CSV.writerow([image_id, final_text])


# A retried image may already have a blank row in the CSV. Move the latest valid
# result into its original row and remove the appended duplicate, preserving the
# original image order used by the aggregation script.
labelled_df = pd.read_csv(destination_path, delimiter=";", dtype={'image_id': object, 'level': object})
valid_level = labelled_df["level"].notna() & labelled_df["level"].astype(str).str.strip().str.lower().ne("nan")
latest_valid = labelled_df[valid_level].drop_duplicates("image_id", keep="last").set_index("image_id")["level"]

for image_id, level in latest_valid.items():
  first_index = labelled_df.index[labelled_df["image_id"] == image_id][0]
  labelled_df.at[first_index, "level"] = level

labelled_df = labelled_df.drop_duplicates("image_id", keep="first")
labelled_df.to_csv(destination_path, sep=";", index=False)
print(f"Annotations finished, saved to {destination_path}")

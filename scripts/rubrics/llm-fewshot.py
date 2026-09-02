import os
import requests
import base64
import csv
import pandas as pd
import re
import sys
from pathlib import Path


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

# Configuration
if len(sys.argv) != 4:
    print("Usage: python script_name.py <dataset> <task> <version>")
    sys.exit(1)

dataset = str(sys.argv[1])
max_samples = int(os.environ.get("MAX_SAMPLES", 50))
max_label_attempts = int(os.environ.get("MAX_LABEL_ATTEMPTS", 5))
task_to_evaluate = str(sys.argv[2])

if task_to_evaluate == "localization":
   task = "localization"
   task_definition = "Localization consists of determining the position of the objects in a given image, i.e. generating a rectangular bounding box that tightly frames each detected object"

elif task_to_evaluate == "detection":
   task = "object detection"
   task_definition = "Object detection consists of determining the position of the objects in a given image, i.e. generating a rectangular bounding box that tightly frames each detected object and then establishing which of the available categories each one belongs to"

else:
   print("Not implemented")
   sys.exit(1)



version = int(sys.argv[3])

versions_available = [16]

if version not in versions_available:
   print("Version not available")
   sys.exit(1)

MODEL_ID = "DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF"
MODEL_NAME = "qwen3.6-27b-q6"
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
IMAGE_DIRS = {"coco-2017": Path.home() / "fiftyone/coco-2017/validation/data","voc-2007": Path.home() / "fiftyone/voc-2007/validation/data","driving": Path("../vision_datasets/driving-validation"),}

image_ids_path = os.environ.get("IMAGE_IDS_PATH", f"./outputs/object_detection/images_experiment_{dataset}.csv")
image_ids = pd.read_csv(image_ids_path, dtype={'image_id': object})

# check if those instances have been already labelled
annotations_dir = Path(os.environ.get("ANNOTATIONS_DIR", "./outputs/annotations"))
os.makedirs(annotations_dir, exist_ok=True)
destination_path = annotations_dir / f"v{version}_{task_to_evaluate}_fewshot_labelled_images_{dataset}_{MODEL_NAME}.csv"
labelled_prev = False
already_labelled = []
if os.path.isfile(destination_path):
  labelled_prev = True
  already_labelled_df = pd.read_csv(destination_path, delimiter=";", dtype={'image_id': object, 'level': object})

  for i,row in already_labelled_df.iterrows():
    if pd.notna(row["level"]) and str(row["level"]).strip().lower() != "nan":
      already_labelled.append(row["image_id"])


few_shot_dir = Path("data/prompts/images")
prompt_path = Path("data/prompts/few_shot_prompt_1dim_rubric.txt")
prompt_template = prompt_path.read_text(encoding="utf-8")
prompt_template = prompt_template.replace("[TASK_DEFINITION]", task_definition).replace("[TASK]", task)
system_prompt, user_prompt = prompt_template.split("[TARGET_IMAGE]", maxsplit=1)
system_content = build_prompt_content(system_prompt, few_shot_dir)


with open(destination_path, 'a', newline='', encoding='utf-8') as CSV_file:
    writer_CSV = csv.writer(CSV_file, delimiter=';')

    if not labelled_prev:
      writer_CSV.writerow(['image_id', 'level'])

    for i, row in image_ids.iterrows():
        
        if i == max_samples:
           print("Max Samples reached")
           break
        image_id = str(row["image_id"])
        print(image_id)

        if image_id in already_labelled:
          print("Labelled!")
          continue

        IMAGE_PATH = next(IMAGE_DIRS[dataset].glob(f"{image_id}.*"))
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

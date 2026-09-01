import os
import requests
import base64
import csv
import pandas as pd
import sys
from pathlib import Path

# Configuration
if len(sys.argv) != 4:
    print("Usage: python script_name.py <dataset> <task> <version>")
    sys.exit(1)

dataset = str(sys.argv[1])
max_samples = int(os.environ.get("MAX_SAMPLES", 50))
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
IMAGE_DIRS = {
  "coco-2017": Path.home() / "fiftyone/coco-2017/validation/data",
  "voc-2007": Path.home() / "fiftyone/voc-2007/validation/data",
  "driving": Path("../vision_datasets/driving-validation"),
}
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
  already_labelled_df = pd.read_csv(destination_path, delimiter=";",dtype={'image_id': object})

  for i,row in already_labelled_df.iterrows():
    already_labelled.append(row["image_id"])


few_shot_dir = Path("data/prompts/imagenes")
few_shot_1 = base64.b64encode(open(few_shot_dir / "000511_level1.jpg", 'rb').read()).decode('ascii')
few_shot_1b = base64.b64encode(open(few_shot_dir / "000000110359_level1.jpg", 'rb').read()).decode('ascii')
few_shot_2 = base64.b64encode(open(few_shot_dir / "000913_level2.jpg", 'rb').read()).decode('ascii')
few_shot_2b = base64.b64encode(open(few_shot_dir / "000000065736_level2.jpg", 'rb').read()).decode('ascii')
few_shot_3 = base64.b64encode(open(few_shot_dir / "000575_level3.jpg", 'rb').read()).decode('ascii')
few_shot_3b = base64.b64encode(open(few_shot_dir / "000853_level3.jpg", 'rb').read()).decode('ascii')
few_shot_4 = base64.b64encode(open(few_shot_dir / "000377_level4.jpg", 'rb').read()).decode('ascii')
few_shot_4b = base64.b64encode(open(few_shot_dir / "000000036494_level4.jpg", 'rb').read()).decode('ascii')


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
        encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
        headers = {"Content-Type": "application/json"}

        if version == 16:
            rubric = "Below you have a detailed description of each level's requirements. Read them carefully and select the most suitable one.\n\nLevel 5. At this level, the system requires capabilities to process and interpret a complex scene in real-time, recognize and categorize objects with near-perfect accuracy, or understand context with nuanced detail. It might require constructing 3D interpretations from monocular 2D vision using background knowledge or handling deformable objects with a very high degree of robustness or extraordinary precision. It may contain too many objects at the same time or degrees of blur or lighting that are extremely hard, with objects that are very domain-specific or very similar between them.\n\nLevel 4. At this level, the system needs vision to succeed at everyday tasks humans face, reliably interpreting the scene. Robustness is needed so that the scale, the orientation, or the environment do not affect accuracy. Scenes can be very complex, including many objects, which might be partially hidden or overlapped. Images have a resolution, uneven lighting, or blur in such a way that makes perception very hard. Even if the scene might seem simple, objects are positioned at an unusual angle or captured from a strange perspective, making recognition difficult.\n\nLevel 3. At this level, interpreting the image requires precise analysis, and reliable pattern recognition to identify and categorize objects accurately. This level does not include complex scenes with many details, poor lighting conditions, or atypical objects or event types, but slight blur or medium resolution is accepted, or heavy distortions and noise in parts that do not affect the objects to be recognized, or slight changes in appearance or perspective from the usual ones.\n\nLevel 2. This level requires the respondents to have basic capabilities, there are no more than a couple of objects in the scene, and they are presented in usual presentations in appearance, scale, or context. Images may include a close-up of an object, or they have an appropriate resolution with minimal blur or have good lighting with minor shadows or occlusions affecting the relevant objects in the image.\n\nLevel 1. This level includes simple vision tasks. They require distinguishing between common shapes, detecting color, or identifying large, distinct objects within a limited scope and clear backgrounds, with no occlusion. Images at this level don't require understanding context or making nuanced distinctions between objects or features. The resolution is sufficient for the size of the objects in question, and the image does not have significant blur, occlusions, or bad lighting conditions affecting the recognition task.\n\n"

        # Payload for the request
        payload = {
        "model": MODEL_ID,
        "messages": [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": f"You are going to score the level of {task} [1, 2, 3, 4, 5] required to interpret the image you will be shown. {task_definition}. 1 represents very easy images and 5 represents very difficult images.\n\n",
                },
                {
                "type": "text",
                "text": rubric,
                },
                {
                "type": "text",
                "text": f"I will first give you a few examples to illustrate it (as few-shot learning). Then you will have to determine the level of the new image. Please *only* answer giving a natural number between 1 (very easy {task} problem) and 5 (very difficult {task} problem) as the level you think better represents the {task} demands of the image."
                
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_2}"
                }
                },
                {
                "type": "text",
                "text": "Level: 2"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_4}"
                }
                },
                {
                "type": "text",
                "text": "Level: 4"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_3}"
                }
                },
                {
                "type": "text",
                "text": "Level: 3"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_1}"
                }
                },
                {
                "type": "text",
                "text": "Level: 1"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_3b}"
                }
                },
                {
                "type": "text",
                "text": "Level: 3"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_2b}"
                }
                },
                {
                "type": "text",
                "text": "Level: 2"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_1b}"
                }
                },
                {
                "type": "text",
                "text": "Level: 1"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{few_shot_4b}"
                }
                },
                {
                "type": "text",
                "text": "Level: 4"
                },
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
                },
                {
                "type": "text",
                "text": "Level: "
                },
            ]
            },
        ],
        "temperature": 0,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": True}
        }
        response = requests.post(MODEL_ENDPOINT, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        final_text = response.json()['choices'][0]["message"]["content"].strip()
        writer_CSV.writerow([image_id, final_text])
        


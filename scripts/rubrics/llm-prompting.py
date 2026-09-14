import os
import requests
import base64
import csv
import json
from jsonschema import Draft202012Validator
import concurrent.futures
import pandas as pd
import re
import sys
import threading
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import (PARTITIONS, PROMPT_STRATEGIES, annotations_dir, load_experiment_config, load_manifest, resolve_repo_path, split_config_for_version,)



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
parser.add_argument("dataset", choices=("coco-2017", "voc-2007", "driving", "coco-rem"))
#parser.add_argument("task", choices=("detection", "localization"))
parser.add_argument("version", type=int)
parser.add_argument("prompt_strategy", choices=PROMPT_STRATEGIES)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--response-format", choices=("level", "json"), default="level", help="Store a single level in CSV (default) or dimension annotations in JSONL",)
parser.add_argument("--max-samples", type=int, default=int(os.environ["MAX_SAMPLES"]) if "MAX_SAMPLES" in os.environ else None, help="Annotate only the first N manifest rows; by default annotate the whole partition",)
parser.add_argument("--max-label-attempts", type=int, default=int(os.environ.get("MAX_LABEL_ATTEMPTS", 5)), )
parser.add_argument("--overwrite", action="store_true", help="Replace this model's existing annotation file instead of resuming it", )
parser.add_argument("--validate-prompt", action="store_true", help="Validate the configured prompt and examples without contacting the model",)
parser.add_argument("--workers", type=int, default=1, help="Concurrent annotation requests; keep at or below the llama-server --parallel slot count, and size -c for workers x per-request context",)
parser.add_argument("--dimension", help="Use one configured separate dimension prompt (JSON zero-shot only)")
args = parser.parse_args()
if args.dimension and (args.response_format != "json" or args.prompt_strategy != "zeroshot"):
  parser.error("--dimension requires --response-format json and zeroshot")

dataset = args.dataset
max_samples = args.max_samples
max_label_attempts = args.max_label_attempts
task_to_evaluate = "detection"

task = "object detection"
task_definition = "Object detection consists of determining the position of the objects in a given image, i.e. generating a rectangular bounding box that tightly frames each detected object and then establishing which of the available categories each one belongs to"

version = args.version
experiment_config = load_experiment_config(version)
split_config = split_config_for_version(version)

MODEL_ID = "bartowski/Qwen3.8-27B-GGUF"
MODEL_NAME = "qwen3.8-27b-q5"
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")

# Reasoning stays enabled for every assessment. On some borderline images the
# model enters an unbounded self-verification loop, exhausts the context window
# and returns no level at all (19 of the first 667 v16 calibration rows, 2.8%).
# The loop is bounded server-side with --reasoning-budget, which closes the
# thinking block and lets the model state its level; see the llama-server command
# in README.md. This client-side cap only bounds the cost of one request, so a
# runaway ends in ~3 minutes instead of consuming the whole window.
# Must exceed the server --reasoning-budget so the level still fits after the
# thinking block is closed; otherwise the answer itself gets truncated.
MAX_REASONING_TOKENS = int(os.environ.get("MAX_REASONING_TOKENS", 12000))
VALID_LEVELS = {"1", "2", "3", "4", "5"}


def parse_level(text):
  """Accept only an unambiguous level, so a cut-off reasoning trace can never be stored as one."""
  text = (text or "").strip()
  if text in VALID_LEVELS:
    return text
  match = re.fullmatch(r"[^\d]*([1-5])[^\d]*", text)
  return match.group(1) if match else None


def reject_json_constant(value):
  raise ValueError(f"Invalid JSON constant: {value}")


def parse_annotation(text):
  if args.response_format == "level":
    return parse_level(text)
  try:
    annotation = json.loads(text or "", parse_constant=reject_json_constant)
  except ValueError:
    return None
  return annotation if annotation_validator.is_valid(annotation) else None


def read_json_annotations(path):
  records = {}
  with open(path, encoding="utf-8") as file:
    for line in file:
      record = json.loads(line)
      if record["image_id"] not in records or record["annotation"] is not None:
        records[record["image_id"]] = record
  return records


def request_annotation(payload):
  response = requests.post(MODEL_ENDPOINT, headers={"Content-Type": "application/json"}, json=payload, timeout=900)
  response.raise_for_status()
  choice = response.json()["choices"][0]
  message = choice["message"]
  return (parse_annotation(message.get("content")), choice.get("finish_reason"), (message.get("reasoning_content") or ""),)
image_ids = load_manifest(args.partition, dataset, split_config)
if max_samples is not None:
  if max_samples < 1:
    parser.error("--max-samples must be positive")
  image_ids = image_ids.iloc[:max_samples]

prompt_key = f"prompt_{args.prompt_strategy}"
if args.response_format == "json":
  prompt_key = "prompt_joint" if args.prompt_strategy == "zeroshot" else "prompt_fewshot_json"
if args.dimension:
  separate_prompts = experiment_config["rubric"]["prompts_separate"]
  if args.dimension not in separate_prompts:
    parser.error(f"Unknown dimension: {args.dimension}; choose from {', '.join(separate_prompts)}")
  prompt_path = resolve_repo_path(separate_prompts[args.dimension])
else:
  if prompt_key not in experiment_config["rubric"]:
    raise ValueError(f"Experiment v{version} does not define rubric.{prompt_key}")
  prompt_path = resolve_repo_path(experiment_config["rubric"][prompt_key])
prompt_template = prompt_path.read_text(encoding="utf-8")
if prompt_template.count("[TARGET_IMAGE]") != 1:
  raise ValueError(f"{prompt_path} must contain exactly one [TARGET_IMAGE] marker")
for marker in ("[TASK]", "[TASK_DEFINITION]"):
  if marker not in prompt_template:
    raise ValueError(f"{prompt_path} must contain the {marker} marker")
prompt_template = prompt_template.replace("[TASK_DEFINITION]", task_definition).replace("[TASK]", task)
system_prompt, user_prompt = prompt_template.split("[TARGET_IMAGE]", maxsplit=1)

image_markers = re.findall(r"\[IMAGE:[^\]]+\]", system_prompt)
if args.prompt_strategy == "zeroshot" and image_markers:
  raise ValueError(f"Zero-shot prompt {prompt_path} must not contain example images")
if args.prompt_strategy == "fewshot" and not image_markers:
  raise ValueError(f"Few-shot prompt {prompt_path} must contain at least one example image")

example_dir = resolve_repo_path(experiment_config["rubric"].get("examples", prompt_path.parent))
system_content = build_prompt_content(system_prompt, example_dir)

if args.response_format == "json":
  schema_path = resolve_repo_path(experiment_config["rubric"]["response_schema"])
  annotation_schema = json.loads(schema_path.read_text(encoding="utf-8"))
  if args.dimension:
    fields = {args.dimension, "target_count_estimate", "target_inventory_uncertain", "depiction_types"}
    for branch in annotation_schema["anyOf"]:
      branch["properties"] = {key: value for key, value in branch["properties"].items() if key in fields}
      branch["required"] = list(branch["properties"])
  Draft202012Validator.check_schema(annotation_schema)
  annotation_validator = Draft202012Validator(annotation_schema)

if args.validate_prompt:
  print(f"Valid {args.prompt_strategy} prompt for v{version}: {prompt_path}")
  sys.exit(0)

# Resume only completed annotations; failed requests remain eligible for retry.
output_annotations_dir = annotations_dir(version, args.partition)
if args.dimension:
  output_annotations_dir = output_annotations_dir / "separate" / args.dimension
os.makedirs(output_annotations_dir, exist_ok=True)
extension = "jsonl" if args.response_format == "json" else "csv"
destination_path = output_annotations_dir / f"v{version}_{task_to_evaluate}_{args.prompt_strategy}_labelled_images_{dataset}_{MODEL_NAME}.{extension}"
if args.overwrite and destination_path.is_file():
  destination_path.unlink()
labelled_prev = destination_path.is_file()
already_labelled = []
if labelled_prev:
  if args.response_format == "json":
    already_labelled = [image_id for image_id, record in read_json_annotations(destination_path).items() if record["annotation"] is not None]
  else:
    already_labelled_df = pd.read_csv(destination_path, delimiter=";", dtype={'image_id': object, 'level': object})
    for i, row in already_labelled_df.iterrows():
      if pd.notna(row["level"]) and str(row["level"]).strip().lower() != "nan":
        already_labelled.append(str(row["image_id"]))


def annotate_image(image_id, image_path):
  if not image_path.is_file():
    raise FileNotFoundError(f"Manifest image not found: {image_path}")
  encoded_image = encode_image(image_path)
  payload = {"model": MODEL_ID, "messages": [{"role": "system","content": system_content,},{"role": "user","content": [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},{"type": "text","text": user_prompt,},]},],"temperature": 0,"top_p": 0.95,"max_tokens": MAX_REASONING_TOKENS,"chat_template_kwargs": {"enable_thinking": True}}
  if args.response_format == "json":
    payload["response_format"] = {"type": "json_schema", "json_schema": {"name": f"rubric_v{version}", "schema": annotation_schema}}
  # Retrying the identical request is worth doing here: reasoning length varies
  # widely between calls even at temperature 0 (750 to 8,550 tokens observed for
  # one image), so a repeat attempt can terminate where the previous one did not.
  for attempt in range(max_label_attempts):
    annotation, finish_reason, _ = request_annotation(payload)
    if annotation is not None:
      return image_id, annotation
    print(f"No valid annotation returned for {image_id} (finish_reason={finish_reason}). Attempt {attempt + 1}/{max_label_attempts}.")
  print(f"Could not obtain a valid annotation for {image_id}; recording it as missing.")
  return image_id, None


pending = [(str(row["image_id"]), Path(row["filepath"])) for _, row in image_ids.iterrows() if str(row["image_id"]) not in already_labelled]
print(f"{len(pending)} of {len(image_ids)} images need annotation, using {args.workers} worker(s)")

with open(destination_path, 'a', newline='', encoding='utf-8') as output_file:
    if args.response_format == "level":
      writer_CSV = csv.writer(output_file, delimiter=';')
      if not labelled_prev:
        writer_CSV.writerow(['image_id', 'level'])
        output_file.flush()

    # Rows are written in completion order rather than manifest order, and flushed
    # immediately so an interrupted run keeps every finished annotation. The dedup
    # pass below still keys on image_id, and every downstream merge joins on it.
    write_lock = threading.Lock()
    completed = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
      futures = [pool.submit(annotate_image, image_id, image_path) for image_id, image_path in pending]
      for future in concurrent.futures.as_completed(futures):
        image_id, annotation = future.result()
        with write_lock:
          if args.response_format == "json":
            output_file.write(json.dumps({"image_id": image_id, "annotation": annotation}, ensure_ascii=False) + "\n")
          else:
            writer_CSV.writerow([image_id, annotation])
          output_file.flush()
          completed += 1
          print(f"{completed}/{len(pending)} {image_id}: {annotation}")
    finally:
      pool.shutdown(cancel_futures=True)


if args.response_format == "json":
  records = read_json_annotations(destination_path)
  with open(destination_path, 'w', encoding='utf-8') as output_file:
    for record in records.values():
      output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
else:
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

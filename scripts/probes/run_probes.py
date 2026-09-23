"""Ask the annotator for target-population statistics that COCO-ReM can verify.

These probes measure whether the model extracts object information from the scene, separately
from whether the rubric turns that information into useful levels. They share the rubric's target
scope wording, so both describe the same population.

python scripts/probes/run_probes.py count --partition calibration --annotator qwen3.8-27b-q5 --workers 4
"""
import argparse
import base64
import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path

import jsonschema
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, REPO_ROOT, load_manifest, load_split_config, load_toml, resolve_repo_path

MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
MAX_REASONING_TOKENS = int(os.environ.get("MAX_REASONING_TOKENS", 12000))

config = load_toml("configs/probes.toml")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("probe", choices=tuple(config["prompts"]))
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--dataset", default="coco-rem")
parser.add_argument("--annotator", required=True, help="Names the output file; one per served model")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--attempts", type=int, default=3)
parser.add_argument("--max-samples", type=int)
parser.add_argument("--validate-prompt", action="store_true")
args = parser.parse_args()

schema = json.loads(resolve_repo_path(config["schemas"]).read_text())[args.probe]
prompt = resolve_repo_path(config["prompts"][args.probe]).read_text()
assert prompt.count("[TARGET_IMAGE]") == 1, f"{args.probe} prompt needs exactly one [TARGET_IMAGE]"
task = "object detection"
task_definition = ("Object detection consists of determining the position of the objects in a given image, "
                   "i.e. generating a rectangular bounding box that tightly frames each detected object and "
                   "then establishing which of the available categories each one belongs to")
prompt = prompt.replace("[TASK_DEFINITION]", task_definition).replace("[TASK]", task)
system_prompt, user_prompt = prompt.split("[TARGET_IMAGE]", maxsplit=1)
if args.validate_prompt:
    print(f"Valid probe prompt: {args.probe}; answer keys {sorted(set(schema['required']))}")
    sys.exit(0)

images = load_manifest(args.partition, args.dataset, load_split_config(config["split_config"]))
if args.max_samples:
    images = images.iloc[:args.max_samples]
output_dir = REPO_ROOT / "outputs/probes" / args.partition
output_dir.mkdir(parents=True, exist_ok=True)
destination = output_dir / f"{args.probe}_{args.dataset}_{args.annotator}.jsonl"

answered = set()
if destination.is_file():
    answered = {json.loads(line)["image_id"] for line in destination.read_text().splitlines()
                if json.loads(line).get("answer") is not None}


def ask(image_id, image_path):
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {"model": args.annotator, "temperature": 0, "top_p": 0.95,
               "max_tokens": MAX_REASONING_TOKENS,
               "chat_template_kwargs": {"enable_thinking": True},
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": args.probe, "strict": True, "schema": schema}},
               "messages": [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                                {"type": "text", "text": user_prompt}]}]}
    for attempt in range(args.attempts):
        response = requests.post(MODEL_ENDPOINT, json=payload, timeout=900)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content") or ""
        try:
            answer = json.loads(content)
            jsonschema.validate(answer, schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as error:
            print(f"{image_id}: invalid answer on attempt {attempt + 1}/{args.attempts} ({type(error).__name__})")
            continue
        return image_id, answer
    return image_id, None


pending = [(row.image_id, Path(row.filepath)) for row in images.itertuples() if row.image_id not in answered]
print(f"{args.probe}: {len(pending)} of {len(images)} images to query with {args.workers} worker(s)")

with destination.open("a", encoding="utf-8") as sink:
    lock = threading.Lock()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(ask, image_id, path) for image_id, path in pending]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            image_id, answer = future.result()
            with lock:
                sink.write(json.dumps({"image_id": image_id, "annotator": args.annotator,
                                       "probe": args.probe, "answer": answer}) + "\n")
                sink.flush()
                print(f"{done}/{len(pending)} {image_id}: {answer}")
    finally:
        pool.shutdown(cancel_futures=True)
print(f"Saved {destination}")

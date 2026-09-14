"""Object-count prompt: the same annotator asked only how many objects are present.

Tests whether the rubric predicts detector performance beyond what counting objects gives,
which is the property most strongly correlated with the detector score. Counts are binned
into five levels so the evaluation treats them exactly like a rubric.

python scripts/baselines/mllm_count.py --partition calibration --workers 4
"""
import argparse
import base64
import concurrent.futures
import csv
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, baselines_dir, load_manifest, load_split_config

PROMPT = """Count the object instances in this image that an object detector would need to find. Object detection consists of determining the position of the objects in a given image, i.e. generating a rectangular bounding box that tightly frames each detected object and then establishing which of the available categories each one belongs to.

Return only the number of such object instances as a natural number. Do not return an explanation, label, prefix, punctuation, or additional text."""

MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
MAX_REASONING_TOKENS = int(os.environ.get("MAX_REASONING_TOKENS", 12000))

NAME="mllm_count"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--attempts", type=int, default=3)
args = parser.parse_args()

config = load_split_config()
images = load_manifest(args.partition, split_config=config).sort_values(["dataset", "image_id"]).reset_index(drop=True)
random_ids = np.flatnonzero(images["selection_group"].eq("random")) if args.partition == "calibration" else np.arange(len(images))
output_dir = baselines_dir(args.partition)
output_dir.mkdir(parents=True, exist_ok=True)
raw_path = output_dir / f"{NAME}_raw.csv"

answered = {}
if raw_path.is_file():
    previous = pd.read_csv(raw_path, dtype={"image_id": str})
    answered = dict(zip(previous["image_id"], previous["value"]))


def ask(image_id, image_path):
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {"model": "baseline", "temperature": 0, "top_p": 0.95, "max_tokens": MAX_REASONING_TOKENS,
               "chat_template_kwargs": {"enable_thinking": True},
               "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                   {"type": "text", "text": PROMPT}]}]}
    for attempt in range(args.attempts):
        response = requests.post(MODEL_ENDPOINT, json=payload, timeout=900)
        response.raise_for_status()
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
        match = re.fullmatch(r"[^\d]*(\d+)[^\d]*", content)
        if match:
            return image_id, int(match.group(1))
        print(f"No count for {image_id}, attempt {attempt + 1}/{args.attempts}")
    return image_id, None


pending = [(row["image_id"], Path(row["filepath"])) for _, row in images.iterrows() if row["image_id"] not in answered]
print(f"{len(pending)} of {len(images)} images to query with {args.workers} worker(s)")

with open(raw_path, "a", newline="", encoding="utf-8") as raw_file:
    writer = csv.writer(raw_file)
    if not answered:
        writer.writerow(["image_id", "value"])
        raw_file.flush()
    lock = threading.Lock()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(ask, image_id, path) for image_id, path in pending]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            image_id, value = future.result()
            with lock:
                writer.writerow([image_id, value])
                raw_file.flush()
                answered[image_id] = value
                print(f"{done}/{len(pending)} {image_id}: {value}")
    finally:
        pool.shutdown(cancel_futures=True)

images["count"] = images["image_id"].map(answered)
images = images[images["count"].notna()].copy()
# Counts act multiplicatively on performance, so the log is the feature to map linearly.
images["log_count"] = np.log1p(images["count"].to_numpy(dtype=float))
output_path = output_dir / f"{NAME}.csv"
images[["dataset", "image_id", "count", "log_count"]].to_csv(output_path, index=False)
print(f"Saved {output_path}: count median {images['count'].median():.0f}, max {images['count'].max():.0f}")

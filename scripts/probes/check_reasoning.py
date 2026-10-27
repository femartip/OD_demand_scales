"""Check that the served model actually reasons, using the probe runner's exact requests.

Sends the same payload as run_probes.py (prompt, schema, sampling, enable_thinking) for the first N
manifest images, saves every raw server response, and reports per image whether a thinking block
came back and how many tokens were generated. Nothing is written to outputs/probes.

python scripts/probes/check_reasoning.py count --partition calibration --images 3 --gate-median 1000 \
    --out reasoning_check.jsonl
"""
import argparse
import base64
import concurrent.futures
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import jsonschema
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, load_manifest, load_split_config, load_toml, resolve_repo_path

MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
MAX_REASONING_TOKENS = int(os.environ.get("MAX_REASONING_TOKENS", 12000))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 900))

config = load_toml("configs/probes.toml")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("probe", choices=tuple(config["prompts"]))
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--dataset", default="coco-rem")
parser.add_argument("--annotator", default="reasoning-check")
parser.add_argument("--images", type=int, default=3)
parser.add_argument("--image-ids", type=Path, help="CSV with an image_id column; overrides --images")
parser.add_argument("--out", type=Path, required=True, help="JSONL of raw responses")
parser.add_argument("--no-schema", action="store_true", help="Drop response_format (diagnostic only)")
parser.add_argument("--gate-median", type=int, help="Exit 1 unless median generated tokens exceed this")
# Off-ladder sampling, diagnostic only: when none of these is given the request is identical to run_probes.py.
parser.add_argument("--temperature", type=float)
parser.add_argument("--top-k", type=int)
parser.add_argument("--min-p", type=float)
parser.add_argument("--presence-penalty", type=float)
# Hosted cross-check through OpenRouter (key from OPENROUTER_API_KEY or ~/.openrouter_key). Same prompt,
# schema and sampling; thinking is requested with OpenRouter's reasoning parameter instead of
# chat_template_kwargs, and the provider chooses the image resolution.
parser.add_argument("--openrouter-model", help="e.g. qwen/qwen3.5-122b-a10b")
parser.add_argument("--provider", help="Pin one OpenRouter provider, no fallbacks (e.g. Novita, Alibaba)")
args = parser.parse_args()
if args.openrouter_model:
    MODEL_ENDPOINT = os.environ.get("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
    # The file may hold the bare key or an assignment such as NAME="sk-or-..."; take the key itself.
    key = os.environ.get("OPENROUTER_API_KEY") or re.search(r"sk-or-[\w-]+", (Path.home() / ".openrouter_key").read_text()).group(0)
    HEADERS = {"Authorization": f"Bearer {key}"}
else:
    HEADERS = {}

# Prompt and payload exactly as in run_probes.py.
schema = json.loads(resolve_repo_path(config["schemas"]).read_text())[args.probe]
prompt = resolve_repo_path(config["prompts"][args.probe]).read_text()
assert prompt.count("[TARGET_IMAGE]") == 1, f"{args.probe} prompt needs exactly one [TARGET_IMAGE]"
task = "object detection"
task_definition = ("Object detection consists of determining the position of the objects in a given image, "
                   "i.e. generating a rectangular bounding box that tightly frames each detected object and "
                   "then establishing which of the available categories each one belongs to")
prompt = prompt.replace("[TASK_DEFINITION]", task_definition).replace("[TASK]", task)
system_prompt, user_prompt = prompt.split("[TARGET_IMAGE]", maxsplit=1)
images = load_manifest(args.partition, args.dataset, load_split_config(config["split_config"]))
if args.image_ids:
    wanted = pd.read_csv(args.image_ids, dtype={"image_id": str}).image_id.tolist()
    images = images.set_index("image_id").loc[wanted].reset_index()
else:
    images = images.iloc[:args.images]


def payload_for(image_path):
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {"model": args.annotator, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
               "max_tokens": MAX_REASONING_TOKENS,
               "chat_template_kwargs": {"enable_thinking": True},
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": args.probe, "strict": True, "schema": schema}},
               "messages": [{"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                                {"type": "text", "text": (system_prompt + user_prompt).strip()}]}]}
    if args.no_schema:
        del payload["response_format"]
    for key, value in (("temperature", args.temperature), ("top_k", args.top_k), ("min_p", args.min_p),
                       ("presence_penalty", args.presence_penalty)):
        if value is not None:
            payload[key] = value
    if args.openrouter_model:
        del payload["chat_template_kwargs"]
        payload["model"] = args.openrouter_model
        # 10,000 mirrors the ladder's --reasoning-budget; providers that cannot cap reasoning ignore it.
        payload["reasoning"] = {"enabled": True, "max_tokens": 10000}
        payload["usage"] = {"include": True}
        if args.provider:
            payload["provider"] = {"order": [args.provider], "allow_fallbacks": False, "require_parameters": True}
    return payload


def post_hosted(image_path):
    """OpenRouter only: retry rate limits, server errors and error bodies; None if it never succeeds."""
    body = {}
    for attempt in range(6):
        response = requests.post(MODEL_ENDPOINT, json=payload_for(image_path), headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if response.status_code in (408, 429, 500, 502, 503, 504):
            body = {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        else:
            response.raise_for_status()
            body = response.json()
            if body.get("choices"):
                return body
        time.sleep(min(15 * 2 ** attempt, 120))
    return {"error": body.get("error", body)}


def ask(image_id, image_path):
    if args.openrouter_model:
        body = post_hosted(image_path)
        if "choices" not in body:
            return {"image_id": image_id, "generated_tokens": None, "thinking_block": False, "reasoning_chars": 0,
                    "valid_answer": False, "finish_reason": None, "error": str(body["error"])[:300], "raw": body}
    else:
        response = requests.post(MODEL_ENDPOINT, json=payload_for(image_path), headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        body = response.json()
    message = body["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    # With --reasoning-format none the thinking stays inside content as <think>...</think>.
    inline = "<think>" in content
    try:
        jsonschema.validate(json.loads(content), schema)
        valid = True
    except (json.JSONDecodeError, jsonschema.ValidationError):
        valid = False
    tokens = (body.get("usage") or {}).get("completion_tokens") or (body.get("timings") or {}).get("predicted_n")
    return {"image_id": image_id, "generated_tokens": tokens, "thinking_block": bool(reasoning.strip()) or inline,
            "reasoning_chars": len(reasoning) if reasoning else (content.find("</think>") if inline else 0),
            "valid_answer": valid, "finish_reason": body["choices"][0].get("finish_reason"), "raw": body}


# Hosted providers rate-limit bursts, so OpenRouter requests go at most 4 at a time.
with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(images)) if args.openrouter_model else len(images)) as pool:
    results = list(pool.map(lambda row: ask(row.image_id, Path(row.filepath)), images.itertuples()))

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as sink:
    for result in results:
        sink.write(json.dumps(result) + "\n")
for r in results:
    print(f"{r['image_id']}: thinking_block={r['thinking_block']} generated_tokens={r['generated_tokens']} "
          f"reasoning_chars={r['reasoning_chars']} valid_answer={r['valid_answer']} finish={r['finish_reason']}"
          + (f" ERROR {r['error']}" if r.get("error") else ""))
errors = sum(1 for r in results if r.get("error"))
if errors:
    print(f"{errors}/{len(results)} requests failed after retries and are left out of the median")
tokens = [r["generated_tokens"] or 0 for r in results if not r.get("error")] or [0]
median = statistics.median(tokens)
print(f"reasoning check ({args.probe}{', no schema' if args.no_schema else ''}): "
      f"{sum(r['thinking_block'] for r in results)}/{len(results)} with a thinking block, "
      f"median generated tokens {median:g}; raw responses -> {args.out}")
if args.gate_median is not None and not median > args.gate_median:
    print(f"GATE FAILED: median generated tokens {median:g} <= {args.gate_median}")
    sys.exit(1)

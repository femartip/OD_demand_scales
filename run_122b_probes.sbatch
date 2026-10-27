#!/bin/bash
#SBATCH --job-name=probe_122b
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
# The batch partition mixes GPU models and SLURM has no GPU types or features to select on, so the
# node is pinned by name. vrhpc2 has A40s with driver 525, which the CUDA 12.0 build targets.
#SBATCH --nodelist=vrhpc2.dsic.upv.es
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/fmarper@upvnet.upv.es/OD_demand_scales/probe_122b_%j.log
#
# Probe battery v2 for qwen3.5-122b-a10b-q5 (Q5_K_M): prompts with the question in the user turn and Qwen's
# thinking-mode sampling (both set in run_probes.py / the prompt files), same server settings as the ladder.
# Resumable: run_probes.py appends to outputs/probes/calibration/<probe>_coco-rem_<annotator>.jsonl
# and skips images that already have an answer, so re-submitting after an interruption continues the run.
# Never delete those files between attempts.
set -euo pipefail

REPO=/data/fmarper@upvnet.upv.es/OD_demand_scales
ANNOTATOR=qwen3.5-122b-a10b-q5-v2
PORT=8080             # preferred; another free port is chosen if something on the node holds it
SERVER_LOG=$REPO/llama_server_122b_${SLURM_JOB_ID:-local}.log
LOAD_TIMEOUT_S=5400   # safety net only; readiness is detected from the log

export PATH=$HOME/build/llama.cpp/build-cuda12.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/cuda-12.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
source "$REPO/.venv/bin/activate"
# sbatch exports the submitting shell's environment; these would silently change the endpoint or
# max_tokens away from the runner defaults every other ladder model used.
unset MODEL_ENDPOINT MAX_REASONING_TOKENS
# Client-side wait per request (not an inference setting). On 4x A40 with 8 slots, answers that use
# the full reasoning budget took up to 887 s, right at the runner's 900 s default.
export REQUEST_TIMEOUT=3600
cd "$REPO"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "ABORT: $*"; exit 1; }

SERVER_PID=""
cleanup() {
    local status=$?
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping llama-server (PID $SERVER_PID)"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$SERVER_PID" 2>/dev/null && { log "llama-server ignored SIGTERM; sending SIGKILL"; kill -KILL "$SERVER_PID" 2>/dev/null || true; }
    fi
    log "Exit status $status"
}
trap cleanup EXIT
trap 'die "received termination signal (scancel or node failure); JSONL progress is kept, re-submit to resume"' TERM INT

server_up() {
    kill -0 "$SERVER_PID" 2>/dev/null && [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" == 200 ]]
}

log "Job ${SLURM_JOB_ID:-local} on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
# llama-server is compiled for sm_80/sm_86 against CUDA 12.0, and the model needs 4x48 GB.
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
# awk, not head: head exits after one line, nvidia-smi then dies of SIGPIPE writing the rest, and
# with pipefail + set -e that kills the job (exit 141).
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk 'NR==1')
[[ "$gpu_names" == "NVIDIA A40" ]] || die "allocated GPUs are '$gpu_names', not A40; pin an A40 node with --nodelist"
# CUDA 12.0 needs driver >= 525. A CUDA 12.8 build failed on this driver with "device kernel image
# is invalid" despite minor-version compatibility, so the toolkit is matched to the driver.
(( ${driver%%.*} >= 525 )) || die "driver $driver is older than 525 and cannot run the CUDA 12.0 build"
command -v llama-server >/dev/null || die "llama-server not found on PATH"
python -c "import sys; print('python', sys.version.split()[0])"

# ---- preflight -----------------------------------------------------------------------------------
# The same literal paths as the llama-server command below, so the check cannot drift from it.
for f in ~/models/qwen3.5-122b-a10b-q5/Qwen3.5-122B-A10B-Q5_K_M.gguf ~/models/qwen3.5-122b-a10b-q5/mmproj-F16.gguf; do
    [[ -s "$f" ]] || die "missing model file $f"
done
python - <<'PY' || die "manifest images do not resolve"
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from common.experiment import load_manifest, load_split_config, load_toml
config = load_toml("configs/probes.toml")
images = load_manifest("calibration", "coco-rem", load_split_config(config["split_config"])).iloc[:400]
missing = [p for p in images.filepath if not Path(p).is_file()]
print(f"manifest: {len(images) - len(missing)}/{len(images)} of the selected images resolve")
sys.exit(1 if missing or len(images) != 400 else 0)
PY
# Nodes are shared, and a port already held by someone else's server would silently answer our
# probes. The port only sets where requests go, never what is asked, so move to a free one.
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if port_busy "$PORT"; then
    log "port $PORT is already in use on $(hostname); choosing another"
    PORT=$((20000 + ${SLURM_JOB_ID:-0} % 10000))
    for _ in $(seq 100); do port_busy "$PORT" || break; PORT=$((PORT + 1)); done
    port_busy "$PORT" && die "no free port found near $PORT on $(hostname)"
fi
export MODEL_ENDPOINT="http://127.0.0.1:$PORT/v1/chat/completions"
log "Serving on port $PORT; MODEL_ENDPOINT=$MODEL_ENDPOINT"

# ---- (a) start llama-server ----------------------------------------------------------------------
log "Starting llama-server; server log: $SERVER_LOG"
llama-server -m ~/models/qwen3.5-122b-a10b-q5/Qwen3.5-122B-A10B-Q5_K_M.gguf \
  --mmproj ~/models/qwen3.5-122b-a10b-q5/mmproj-F16.gguf \
  -ngl 999 -np 8 -c 131072 -fa on --image-min-tokens 1024 \
  --reasoning-budget 10000 \
  --reasoning-budget-message "I have gathered enough evidence. I will now answer." \
  --host 127.0.0.1 --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
log "llama-server PID $SERVER_PID"

# ---- (b) wait for readiness ----------------------------------------------------------------------
start=$SECONDS
until grep -q "listening on" "$SERVER_LOG"; do
    # "failed to" only counts on error-level (E) lines: llama.cpp also logs benign warnings with it,
    # e.g. "W common_fit_params: failed to measure the memory of the extra model" for the mmproj.
    # "failed to fit params" stays fatal at any level: it means the model does not fit the GPUs.
    # "no usable GPU" is only a plain warning, after which llama.cpp silently runs on the CPU.
    if error_line=$(grep -m1 -iE "error loading|out of memory|CUDA error|couldn't bind|failed to fit params|no usable GPU|^[0-9.]+ E .*failed to" "$SERVER_LOG"); then
        tail -n 40 "$SERVER_LOG"
        die "llama-server failed to start: $error_line"
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 40 "$SERVER_LOG"; die "llama-server exited during startup"; }
    (( SECONDS - start < LOAD_TIMEOUT_S )) || { tail -n 40 "$SERVER_LOG"; die "llama-server not ready after ${LOAD_TIMEOUT_S}s"; }
    sleep 5
done
server_up || { tail -n 40 "$SERVER_LOG"; die "llama-server logged 'listening on' but /health is not 200"; }
# The weights alone are ~85 GiB; anything far below that means they are not on the GPUs.
gpu_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$SERVER_PID" '$1==p {s+=$2} END {print s+0}')
log "llama-server holds ${gpu_mib} MiB of GPU memory"
(( gpu_mib >= 80000 )) || { nvidia-smi; tail -n 40 "$SERVER_LOG"; die "llama-server holds only ${gpu_mib} MiB of GPU memory; the model is not on the GPUs"; }
log "llama-server ready after $((SECONDS - start))s"
grep -m1 -E "listening on" "$SERVER_LOG"

# ---- reasoning check: 10-image test of the user-turn layout -------------------------------------------
# run_probes.py now sends one user turn (image, then the whole prompt) and no system message. Under v2
# with the instructions in a system message, these same 10 images were 10/10 short (median 201.5 tokens,
# job 154822). Sends run_probes.py's exact request for those 10 images and stops the job if the median is
# still 400 tokens or less (the planning-only mode); otherwise the full battery follows. Nothing here is
# written to outputs/probes; the raw responses go to reasoning_check_122b_v2_<jobid>.jsonl.
log "Reasoning check: median generated tokens must exceed 400"
PYTHONUNBUFFERED=1 python -u -W ignore scripts/probes/check_reasoning.py count --partition calibration \
  --annotator qwen3.5-122b-a10b-q5-v2 --images 10 --gate-median 400 \
  --out "$REPO/reasoning_check_122b_v2_${SLURM_JOB_ID:-local}.jsonl" \
  || die "reasoning check failed: still the short planning-only mode with the user-turn layout"

# ---- (c) smoke test ------------------------------------------------------------------------------
log "Smoke test: count on 3 images"
PYTHONUNBUFFERED=1 python -u scripts/probes/run_probes.py count \
  --partition calibration --annotator qwen3.5-122b-a10b-q5-v2 --workers 8 --max-samples 3 \
  || die "smoke test exited non-zero"
# Checks the JSONL rather than stdout so a resumed job (0 pending) is judged on the earlier answers.
python - <<'PY' || { tail -n 40 "$SERVER_LOG"; die "smoke test answers came back None"; }
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from common.experiment import load_manifest, load_split_config, load_toml
config = load_toml("configs/probes.toml")
ids = list(load_manifest("calibration", "coco-rem", load_split_config(config["split_config"])).image_id[:3])
path = Path("outputs/probes/calibration/count_coco-rem_qwen3.5-122b-a10b-q5-v2.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
answered = {r["image_id"] for r in rows if r.get("answer") is not None}
missing = [i for i in ids if i not in answered]
print(f"smoke test: {len(ids) - len(missing)}/3 answered" + (f"; None for {missing}" if missing else ""))
sys.exit(1 if missing else 0)
PY

# ---- (d) full battery ----------------------------------------------------------------------------
for probe in count small_fraction truncation category_diversity joint; do
  server_up || { tail -n 40 "$SERVER_LOG"; die "llama-server is not listening on $PORT before probe $probe"; }
  log "Probe $probe"
  PYTHONUNBUFFERED=1 python -u scripts/probes/run_probes.py "$probe" \
    --partition calibration --annotator qwen3.5-122b-a10b-q5-v2 \
    --workers 8 --max-samples 400 \
    || { tail -n 40 "$SERVER_LOG"; die "probe $probe exited non-zero"; }
done

# ---- summary (counts only; scoring happens elsewhere) --------------------------------------------
python - <<'PY'
import json
from pathlib import Path
for probe in ("count", "small_fraction", "truncation", "category_diversity", "joint"):
    path = Path(f"outputs/probes/calibration/{probe}_coco-rem_qwen3.5-122b-a10b-q5-v2.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    answered = {r["image_id"] for r in rows if r.get("answer") is not None}
    print(f"{probe}: {len(answered)}/400 answered -> {path}")
PY
log "All five probes finished"

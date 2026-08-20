#!/usr/bin/env bash
# Rent a GPU, LoRA fine-tune on the Nietzsche corpus, bring the adapter home, kill the pod.
# Requires runpodctl >= 2.9.0 for --wait / --terminate-after.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
MODEL="${MODEL:-dbmdz/german-gpt2}"
EPOCHS="${EPOCHS:-3}"
BLOCK="${BLOCK:-512}"
TAG="${TAG:-$(echo "$MODEL" | tr '/' '-')}"
TEMPLATE="runpod-torch-v280"

GPUS=("NVIDIA GeForce RTX 4090" "NVIDIA RTX A4000" "NVIDIA GeForce RTX 3090" "NVIDIA L4")

# Server-side dead-man switch: RunPod terminates the pod at this time no matter what happens
# to this script. The EXIT trap is the fast path; this is the guarantee.
DEADLINE="$(date -u -d '+2 hours' +%Y-%m-%dT%H:%M:%SZ)"

POD_ID="${POD_ID:-}"
cleanup() {
  if [[ -n "$POD_ID" ]]; then
    echo ">> terminating pod $POD_ID"
    runpodctl pod delete "$POD_ID" >/dev/null 2>&1 \
      || echo "!! DELETE FAILED - run: runpodctl pod delete $POD_ID"
  fi
}
trap cleanup EXIT

if [[ -n "$POD_ID" ]]; then
  echo ">> reusing running pod $POD_ID"
  POD_JSON=$(runpodctl pod get "$POD_ID" -o json)
else
  echo ">> creating pod (auto-terminate at $DEADLINE)"
  for gpu in "${GPUS[@]}"; do
    echo "   trying: $gpu"
    # --wait blocks until port 22 answers with an SSH banner and then prints the pod in
    # `pod get` shape, so there is no poll loop and no "RUNNING but not ready" ambiguity.
    if POD_JSON=$(runpodctl pod create \
          --name nietzsche-lora \
          --template-id "$TEMPLATE" \
          --gpu-id "$gpu" --gpu-count 1 \
          --cloud-type SECURE \
          --container-disk-in-gb 40 --volume-in-gb 20 --volume-mount-path /workspace \
          --ports '22/tcp' --ssh \
          --terminate-after "$DEADLINE" \
          --wait --wait-timeout 8m \
          -o json 2>/tmp/rp_err.$$); then
      POD_ID=$(echo "$POD_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
      echo "   ready: $POD_ID on $gpu"
      break
    fi
    # On wait-timeout the pod is KEPT and its id is in the error object -- reap it or it bills.
    stray=$(python3 -c '
import json,sys
try: print(json.load(open(sys.argv[1])).get("id") or "")
except Exception: print("")' /tmp/rp_err.$$ 2>/dev/null || echo "")
    if [[ -n "$stray" ]]; then
      echo "   wait failed, reaping stranded pod $stray"
      runpodctl pod delete "$stray" >/dev/null 2>&1 || true
    fi
    echo "   unavailable: $(head -c 120 /tmp/rp_err.$$ 2>/dev/null)"
  done
  rm -f /tmp/rp_err.$$
fi
[[ -n "$POD_ID" ]] || { echo "!! no GPU available from the fallback chain"; exit 1; }

read -r HOST PORT < <(echo "$POD_JSON" | python3 -c '
import json,sys
d=json.load(sys.stdin); d=d[0] if isinstance(d,list) and d else d
sh=d.get("ssh") or {}
print(sh.get("ip",""), sh.get("port",""))')
[[ -n "$HOST" && -n "$PORT" ]] || { echo "!! pod ready but no ssh endpoint"; echo "$POD_JSON"; exit 1; }
echo ">> ssh root@$HOST -p $PORT"

SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "root@$HOST")
SCP=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

echo ">> waiting for our key to be accepted"
for _ in $(seq 1 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 4; done
"${SSH[@]}" true || { echo "!! ssh banner up but key rejected"; exit 1; }

echo ">> uploading corpus + scripts"
"${SSH[@]}" "mkdir -p /workspace/nz/runpod"
"${SCP[@]}" "$REPO"/*.txt "root@$HOST:/workspace/nz/"
"${SCP[@]}" "$HERE"/data.py "$HERE"/train_lora.py "$HERE"/generate.py "root@$HOST:/workspace/nz/runpod/"

echo ">> installing deps"
# RunPod's torch image ships a PEP 668 externally-managed python; the container is disposable
# so --break-system-packages is correct. Never pipe this -- a pipe masks the exit status.
"${SSH[@]}" "pip install -q --break-system-packages -U 'transformers>=4.44' peft datasets accelerate"

echo ">> verifying deps"
"${SSH[@]}" "python -c \"
import torch, transformers, peft, datasets
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__, '| peft', peft.__version__)
assert torch.cuda.is_available(), 'no CUDA'
\"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

echo ">> training  model=$MODEL epochs=$EPOCHS block=$BLOCK"
"${SSH[@]}" "cd /workspace/nz/runpod && python train_lora.py --model '$MODEL' --data-dir .. --out /workspace/out --epochs $EPOCHS --block-size $BLOCK 2>&1"

echo ">> sampling"
"${SSH[@]}" "cd /workspace/nz/runpod && python generate.py --model '$MODEL' --adapter /workspace/out 2>&1" \
  | tee "$HERE/samples-$TAG.txt"

echo ">> downloading adapter"
mkdir -p "$HERE/adapters/$TAG"
"${SCP[@]}" -r "root@$HOST:/workspace/out/*" "$HERE/adapters/$TAG/"

echo ">> done. adapter in runpod/adapters/$TAG, samples in runpod/samples-$TAG.txt"

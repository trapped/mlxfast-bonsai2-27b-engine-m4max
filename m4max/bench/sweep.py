"""Sweep speculative decoder/depth directly against bench-worker (no HTTP). Greedy.
Usage: uv run python bench/sweep.py dflash 0,3,7,11,15   |   ... mtp 1,2,3,5"""
import json, os, subprocess, sys, time
from transformers import AutoTokenizer
U = ".."
tok = AutoTokenizer.from_pretrained(f"{U}/reference_weights/Ternary-Bonsai-2-27B-mlx-2bit")
mode, depths = sys.argv[1], [int(d) for d in sys.argv[2].split(",")]
drafter = {"dflash": "Qwen3.8-27B-DFlash2", "mtp": "Qwen3.8-27B-MTP-4bit"}[mode]
prompts = {
  "code": "Write a Python class `LRUCache` with get/put in O(1) using OrderedDict, plus pytest tests.",
  "prose": "Explain how TCP congestion control works (slow start, AIMD, fast retransmit).",
}
p = subprocess.Popen([f"{U}/.build/release/bench-worker", "runtime-worker", "--weights", f"{U}/weights",
     "--drafter", f"{U}/reference_weights/{drafter}", "--speculative-protocol", "v1.1"],
     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/sweep-worker.err","w"), text=True, bufsize=1)
json.loads(p.stdout.readline())
def call(**k):
    p.stdin.write(json.dumps(k) + "\n"); p.stdin.flush(); r = json.loads(p.stdout.readline())
    assert r["ok"], r; return r
N = int(os.environ.get("SWEEP_N", 256))
for d in depths:
  for name, pr in prompts.items():
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": pr}], add_generation_prompt=True,
                     tokenize=False, enable_thinking=False), add_special_tokens=False)
    spec = {"mode": "serial"} if d == 0 else {"mode": mode, mode: {"depth": d}}
    call(id=0, kind="phase_diagnostics")
    call(id=1, kind="free_decode_begin", seed_tokens=ids, spec=spec); call(id=2, kind="free_decode_run", count=16)  # warm
    call(id=0, kind="phase_diagnostics")
    t = time.perf_counter(); call(id=3, kind="free_decode_begin", seed_tokens=ids, spec=spec); ttft = time.perf_counter() - t
    t = time.perf_counter(); r = call(id=4, kind="free_decode_run", count=N); dt = time.perf_counter() - t
    rss = int(subprocess.run(["ps","-o","rss=","-p",str(p.pid)],capture_output=True,text=True).stdout)/2**20
    dr, ac = r.get("drafted_total", 0), r.get("accepted_total", 0)
    al = r.get("acceptance_lengths") or []
    print(json.dumps({"mode": mode if d else "serial", "depth": d, "prompt": name, "decode_tok_s": round(N / dt, 1),
          "ttft_s": round(ttft, 3), "rss_gb": round(rss, 1), "accept_rate": round(ac / dr, 3) if dr else None,
          "mean_committed_per_round": round(sum(al) / len(al), 2) if al else None}), flush=True)

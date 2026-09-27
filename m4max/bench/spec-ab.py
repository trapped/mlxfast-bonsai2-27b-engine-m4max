"""Speculative decode throughput via bonsai-serve: decoder/depth x prompts, 256 greedy tokens.
usage: uv run python bench/spec-ab.py dflash 3,7   (env knobs pass through)"""
import json, os, subprocess, sys
from transformers import AutoTokenizer
U = os.environ.get("BONSAI_UPSTREAM", "..")
tok = AutoTokenizer.from_pretrained(f"{U}/reference_weights/Ternary-Bonsai-2-27B-mlx-2bit")
dec, depths = sys.argv[1], [int(d) for d in sys.argv[2].split(",")]
cmd = [f"{U}/.build-worker/release/bonsai-serve", "--weights", f"{U}/weights"]
if dec != "serial": cmd += ["--drafter", os.environ.get("BONSAI_DRAFTER", f"{U}/reference_weights/Qwen3.8-27B-DFlash2")]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdout.readline()
prompts = {"code": "Write a Python class `LRUCache` with get/put in O(1) using OrderedDict, plus pytest tests.",
           "prose": "Explain how TCP congestion control works (slow start, AIMD, fast retransmit)."}
def gen(ids, n, d):
    p.stdin.write(json.dumps({"prompt_tokens": ids, "max_tokens": n, "stop_tokens": [], "decoder": dec, "depth": d}) + "\n"); p.stdin.flush()
    out = []
    while True:
        r = json.loads(p.stdout.readline())
        if r["event"] == "delta": out += r["tokens"]
        else: return out, r
for d in depths:
    for name, pr in prompts.items():
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": pr}], add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False)
        gen(ids, 16, d)
        out, r = gen(ids, 256, d)
        dt = (r["total_ms"] - r["ttft_ms"]) / 1000
        print(json.dumps({"env": os.environ.get("AB_LABEL", ""), "decoder": dec, "depth": d, "prompt": name,
                          "decode_tok_s": round((len(out) - 1) / dt, 1),
                          "accept_rate": round(r["accepted"] / r["drafted"], 3) if r["drafted"] else None,
                          "tok_per_round": round(len(out) / r["rounds"], 2) if r["rounds"] else None,
                          "tokens_hash": hash(tuple(out)) % 10**8}), flush=True)

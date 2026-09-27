"""Agent-edit style benchmark: the model re-emits a file from its prompt with a small change.
Exercises prompt-lookup drafting. usage: uv run python bench/quote-ab.py dflash 3"""
import json, os, subprocess, sys
from transformers import AutoTokenizer
U = os.environ.get("BONSAI_UPSTREAM", "..")
tok = AutoTokenizer.from_pretrained(f"{U}/reference_weights/Ternary-Bonsai-2-27B-mlx-2bit")
dec, depth = sys.argv[1], int(sys.argv[2])
cmd = [f"{U}/.build-worker/release/bonsai-serve", "--weights", f"{U}/weights"]
if dec != "serial":
    cmd += ["--drafter", os.environ.get("BONSAI_DRAFTER", os.path.abspath("../reference_weights/Qwen3.8-27B-DFlash2"))]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdout.readline()
src = open("bench/bench.py").read()
msg = ("Here is bench/bench.py:\n```python\n" + src + "\n```\n"
       "Rename the function `worker_rss_gb` to `worker_memory_gb` everywhere and output the COMPLETE "
       "updated file in one python code block, with no other text.")
ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                 tokenize=False, enable_thinking=False), add_special_tokens=False)
def gen(n):
    p.stdin.write(json.dumps({"prompt_tokens": ids, "max_tokens": n, "stop_tokens": [248046], "decoder": dec, "depth": depth}) + "\n"); p.stdin.flush()
    out = []
    while True:
        r = json.loads(p.stdout.readline())
        if r["event"] == "delta": out += r["tokens"]
        else: return out, r
gen(8)
out, r = gen(1500)
dt = (r["total_ms"] - r["ttft_ms"]) / 1000
print(json.dumps({"tip": U, "decoder": dec, "depth": depth, "prompt_tokens": len(ids), "gen_tokens": len(out),
                  "decode_tok_s": round((len(out) - 1) / dt, 1),
                  "tok_per_round": round(len(out) / r["rounds"], 2) if r["rounds"] else None,
                  "tokens_hash": hash(tuple(out)) % 10**8}))

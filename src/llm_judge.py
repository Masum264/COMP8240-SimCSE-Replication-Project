"""LLM-as-judge for sentence-pair similarity, on the same 0-4 scale given to human raters.

Backends
  hf         local open model (default Qwen/Qwen2.5-7B-Instruct; add --load_4bit on a 16 GB GPU).
             Reads the probability of the answer digits 0-4 in ONE forward pass (no sampling),
             so results are deterministic. Primary score = expected value over the digits.
  anthropic  Claude API (needs ANTHROPIC_API_KEY; integer answer; check current API docs
             for supported parameters).  Not tested by the author of this script.
  dummy      word-overlap stand-in, ONLY for testing the pipeline without a GPU.

Each pair is scored in both sentence orders (A,B) and (B,A) and averaged, unless
--single_order is given. Output is resumable (safe to re-run after a Colab reset).

  python src/llm_judge.py --input data/eval/own.csv --out results/llm/own.csv --load_4bit
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd

PROMPT_VERSION = "v1"
SYSTEM = "You are an expert annotator rating how similar in meaning two sentences are."
RUBRIC = (
    "Rate how similar the MEANING of the two sentences is, on this scale:\n"
    "4 = same meaning (they say essentially the same thing)\n"
    "3 = mostly similar (similar meaning, but some important information differs)\n"
    "2 = somewhat related (they share some details or a sub-topic, but are not equivalent)\n"
    "1 = barely related (same broad topic or domain only)\n"
    "0 = unrelated (different topics)\n"
    "Judge meaning, not word overlap. Ignore formatting differences."
)
USER_TMPL = (RUBRIC + "\n\nSentence A: {s1}\nSentence B: {s2}\n\n"
             "Answer with a single digit from 0 to 4 and nothing else.")
MAX_CHARS = 1500


def build_user_prompt(s1, s2):
    return USER_TMPL.format(s1=str(s1)[:MAX_CHARS], s2=str(s2)[:MAX_CHARS])


def parse_digit(text):
    """First digit 0-4 in a model answer, or None."""
    m = re.search(r"[0-4]", str(text))
    return int(m.group()) if m else None


class DummyBackend:
    name = "dummy"

    def score(self, s1, s2):
        a, b = set(str(s1).lower().split()), set(str(s2).lower().split())
        j = len(a & b) / max(1, len(a | b))
        return {"exp": 4 * j, "int": int(round(4 * j)), "raw": f"dummy:{j:.3f}"}


class HFBackend:
    def __init__(self, model_name, load_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.name = torch, model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kw = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, **kw).eval()
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, **kw).eval()
        ids = [self.tok.encode(str(d), add_special_tokens=False) for d in range(5)]
        assert all(len(i) == 1 for i in ids), "digits are not single tokens for this tokenizer"
        self.digit_ids = [i[0] for i in ids]

    def score(self, s1, s2):
        torch = self.torch
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(s1, s2)}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        full = torch.softmax(logits, dim=-1)
        mass = float(full[self.digit_ids].sum())
        p = torch.softmax(logits[self.digit_ids], dim=-1).cpu().numpy()
        return {"exp": float((p * np.arange(5)).sum()), "int": int(p.argmax()),
                "raw": json.dumps({"p": [round(float(x), 4) for x in p], "digit_mass": round(mass, 4)})}


class AnthropicBackend:
    def __init__(self, model_name):
        import anthropic
        self.client, self.name = anthropic.Anthropic(), model_name

    def score(self, s1, s2):
        for attempt in range(5):
            try:
                msg = self.client.messages.create(
                    model=self.name, max_tokens=5, temperature=0, system=SYSTEM,
                    messages=[{"role": "user", "content": build_user_prompt(s1, s2)}])
                text = msg.content[0].text.strip()
                d = parse_digit(text)
                return {"exp": float(d) if d is not None else np.nan,
                        "int": d if d is not None else -1, "raw": text}
            except Exception as e:  # noqa: BLE001
                print("API error:", e)
                time.sleep(2 ** attempt)
        return {"exp": np.nan, "int": -1, "raw": "ERROR"}


def make_backend(args):
    if args.backend == "dummy":
        return DummyBackend()
    if args.backend == "anthropic":
        return AnthropicBackend(args.model)
    return HFBackend(args.model, args.load_4bit)


def judge_row(backend, s1, s2, both_orders=True):
    ab = backend.score(s1, s2)
    if not both_orders:
        return {"llm_score": ab["exp"], "llm_ab": ab["exp"], "llm_ba": np.nan,
                "raw_ab": ab["raw"], "raw_ba": ""}
    ba = backend.score(s2, s1)
    return {"llm_score": (ab["exp"] + ba["exp"]) / 2, "llm_ab": ab["exp"], "llm_ba": ba["exp"],
            "raw_ab": ab["raw"], "raw_ba": ba["raw"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["hf", "anthropic", "dummy"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--single_order", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = pd.read_csv(args.out).to_dict("records") if os.path.exists(args.out) else []
    done = {str(r["pair_id"]) for r in rows}
    backend = make_backend(args)
    for _, r in df.iterrows():
        if str(r["pair_id"]) in done:
            continue
        rows.append({"pair_id": r["pair_id"],
                     **judge_row(backend, r["sentence1"], r["sentence2"], not args.single_order)})
        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"{len(rows)}/{len(df)} done")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    meta = {"backend": args.backend, "model": backend.name, "load_4bit": args.load_4bit,
            "both_orders": not args.single_order, "prompt_version": PROMPT_VERSION,
            "prompt_sha": hashlib.sha256((SYSTEM + USER_TMPL).encode()).hexdigest()[:12],
            "date": datetime.date.today().isoformat(), "n": len(rows)}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    print("done:", meta)


if __name__ == "__main__":
    main()

"""Sample from the LoRA-tuned model next to the untouched base, so the delta is visible."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Der Übermensch ist",
    "Was mich nicht umbringt,",
    "Die Moral ist",
    "Und Zarathustra sprach also zum Volke:",
]


def sample(model, tok, prompt, n_tokens, seed):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dbmdz/german-gpt2")
    p.add_argument("--adapter", default="out")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    base.eval()
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev),
        args.adapter,
    )
    tuned.eval()

    for prompt in PROMPTS:
        print("\n" + "=" * 78)
        print(f"PROMPT: {prompt!r}")
        for label, m in (("BASE ", base), ("TUNED", tuned)):
            print("-" * 78)
            print(f"[{label}] {sample(m, tok, prompt, args.tokens, args.seed)}")


if __name__ == "__main__":
    main()

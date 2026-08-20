"""LoRA fine-tune a small causal LM on the Nietzsche corpus.

Style continuation: plain causal LM over fixed-size token blocks, no instruction format.
Blocks never straddle two works; the last 2% of each work is held out for validation so
eval covers all seven books rather than just the last one.
"""

import argparse
import inspect
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from data import load_works

# GPT-2 fuses Q/K/V into a single Conv1D named c_attn; llama-likes keep them separate.
TARGETS = {
    "gpt2": ["c_attn"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def build_splits(works, tok, block_size, val_frac):
    train_blocks, val_blocks = [], []
    for name, text in works:
        ids = tok(text, return_attention_mask=False)["input_ids"]
        blocks = [
            ids[i : i + block_size]
            for i in range(0, len(ids) - block_size + 1, block_size)
        ]
        n_val = max(1, int(len(blocks) * val_frac))
        train_blocks += blocks[:-n_val]
        val_blocks += blocks[-n_val:]
        print(f"  {name:30} {len(ids):>8} tok -> {len(blocks):>5} blocks ({n_val} val)")
    return train_blocks, val_blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dbmdz/german-gpt2")
    p.add_argument("--data-dir", default="..")
    p.add_argument("--out", default="out")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=auto)")
    p.add_argument("--max-train-blocks", type=int, default=0,
                   help="cap training blocks to fit a time budget (0=use all)")
    args = p.parse_args()

    print(f"== loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Tiger Lake has AVX-512 but no AMX, so bf16 would be emulated and slower than fp32.
    # Only take bf16 when there is a GPU actually accelerating it.
    cuda = torch.cuda.is_available()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"== device={'cuda' if cuda else 'cpu'} dtype={'bf16' if cuda else 'fp32'} "
          f"threads={torch.get_num_threads()}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if cuda else torch.float32
    )
    model.config.use_cache = False

    mtype = model.config.model_type
    targets = TARGETS.get(mtype)
    if targets is None:
        raise SystemExit(f"no LoRA target_modules mapped for model_type={mtype!r}")

    print(f"== tokenizing corpus (block_size={args.block_size})")
    works = load_works(Path(args.data_dir))
    train_blocks, val_blocks = build_splits(works, tok, args.block_size, args.val_frac)
    print(f"   train={len(train_blocks)} blocks  val={len(val_blocks)} blocks"
          f"  ({len(train_blocks) * args.block_size / 1e6:.2f}M train tokens)")

    if args.max_train_blocks and len(train_blocks) > args.max_train_blocks:
        # Evenly spaced subsample so the cap still spans all seven works rather than
        # truncating to whichever books happen to come first.
        step = len(train_blocks) / args.max_train_blocks
        train_blocks = [train_blocks[int(i * step)] for i in range(args.max_train_blocks)]
        print(f"   capped to {len(train_blocks)} blocks "
              f"({len(train_blocks) * args.block_size / 1e3:.0f}k tokens)")

    train_ds = Dataset.from_dict({"input_ids": train_blocks})
    val_ds = Dataset.from_dict({"input_ids": val_blocks})

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    model.print_trainable_parameters()

    # transformers 5.x trimmed TrainingArguments (warmup_ratio, among others, is gone) and
    # renamed evaluation_strategy -> eval_strategy somewhere in 4.x. Filter against the real
    # signature so this runs on whatever the image happens to ship.
    wanted = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=cuda,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
    )
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        wanted["evaluation_strategy"] = wanted.pop("eval_strategy")
    if "warmup_ratio" not in supported and "warmup_steps" in supported:
        wanted.pop("warmup_ratio", None)
        wanted["warmup_steps"] = 20
    dropped = sorted(k for k in wanted if k not in supported)
    if dropped:
        print(f"== TrainingArguments: dropping unsupported {dropped} "
              f"(transformers {__import__('transformers').__version__})")
    ta = TrainingArguments(**{k: v for k, v in wanted.items() if k in supported})

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    base_eval = trainer.evaluate()
    print(f"== baseline (untrained adapter) ppl = {math.exp(base_eval['eval_loss']):.2f}")

    trainer.train()

    final_eval = trainer.evaluate()
    print(f"== final ppl = {math.exp(final_eval['eval_loss']):.2f} "
          f"(from {math.exp(base_eval['eval_loss']):.2f})")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"== adapter saved to {args.out}")


if __name__ == "__main__":
    main()

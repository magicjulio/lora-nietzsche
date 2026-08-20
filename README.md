# lora fine tune on nietzsche corpus

i gathered most of the works from the german philosopher friedrich nietzsche. 

base model: dbmdz/german-gpt2, 124M parameters.

Two importent files:
runpod/train_lora.py: Tokenize → chunk → LoRA train → save adapter.
runpod/generate.py: Load base + adapter, sample continuations.

val perplexity 74.48 → 54.44 (−27%)
3 epochs, 255 steps, ~92s on an RTX A4000
0.70M tokens, 1359 train blocks + 23 val, 512-token blocks
LoRA r=16, alpha=32, target c_attn — 589,824 / 125M params trainable (0.47%)


usage:
'''
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
m = PeftModel.from_pretrained(m, "runpod/adapters/dbmdz-german-gpt2")
'''

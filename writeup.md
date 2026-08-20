# lora fine tune on nietzsche corpus

i gathered most of the works and even some of the letters 
from the german philosopher friedrich nietzsche. 

base model: dbmdz/german-gpt2, 124M parameters.

Two importent files:
runpod/train_lora.py: Tokenize → chunk → LoRA train → save adapter.
runpod/generate.py: Load base + adapter, sample continuations.

used an RTX 4090 from runpod.

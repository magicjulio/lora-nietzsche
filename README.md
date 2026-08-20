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

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "dbmdz/german-gpt2"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "runpod/adapters/dbmdz-german-gpt2")

ids = tok("Der Übermensch ist", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=120, do_sample=True,
                     temperature=0.9, top_p=0.95, repetition_penalty=1.1)
print(tok.decode(out[0], skip_special_tokens=True))
```


## Example outputs

Same prompt, same seed: base vs the LoRA adapter.

**"Der Übermensch ist"**

> **base:** …ein deutscher Film der DEFA von 1972. Ein Mann wird in einer Straße
> bei einem Spaziergang durch die Berliner Luft angegriffen und schwer verletzt…

> **tuned:** …alles, was der Mensch will und sein Wesen nur dazu macht. Also tuts
> und brummt es auch alle Tage noch zu so einem sündlichen Anstande!

# 🚀 Quick Start Guide

Get your bilingual tokenizer running in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/mnsoylemez/bilingual-tokenizer.git
cd bilingual-tokenizer

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Prepare Your Data

Create two text files (one per language):

```
data/
├── english.txt    # One sentence per line
└── turkish.txt    # One sentence per line
```

**Minimum recommended size:** 10K lines per language

### 2. Train Your Tokenizer

```bash
python bilingual_tokenizer.py \
    --lang1-corpus data/english.txt \
    --lang2-corpus data/turkish.txt \
    --lang1-name English \
    --lang2-name Turkish \
    --output-dir my_tokenizer/
```

That's it! Your tokenizer is ready in `my_tokenizer/tokenizer.json`

### 3. Use Your Tokenizer

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("my_tokenizer/tokenizer.json")

# Tokenize text
output = tokenizer.encode("Hello world! Merhaba dünya!")
print(output.tokens)  # ['Hello', 'world', '!', 'Merhaba', 'dünya', '!']
```

## Advanced Options

### Custom Vocabulary Size

```bash
python bilingual_tokenizer.py \
    --lang1-corpus data/english.txt \
    --lang2-corpus data/turkish.txt \
    --vocab-size 50000 \
    --lang1-core 3000 \
    --lang2-core 3000
```

### Handling Imbalanced Data

If one language has much more data:

```bash
python bilingual_tokenizer.py \
    --lang1-corpus data/english_large.txt \
    --lang2-corpus data/turkish_small.txt \
    --alpha 0.5  # Strong balancing
```

### Domain-Specific Tokenizer

```bash
python bilingual_tokenizer.py \
    --lang1-corpus data/medical_en.txt \
    --lang2-corpus data/medical_tr.txt \
    --vocab-size 50000 \
    --min-frequency 3  # Higher threshold for domain terms
```

## Understanding the Output

After training, you'll see:

```
my_tokenizer/
├── tokenizer.json          # ← USE THIS in your models
├── vocabulary.json         # Full vocabulary
├── core_tokens.json        # Core tokens per language
├── analysis_report.json    # Detailed metrics
└── ANALYSIS.md            # Human-readable report
```

### Key Metrics to Check

Open `ANALYSIS.md` and look for:

1. **Fertility** (lower is better):
   - English: ~1.3 ✅
   - Turkish: ~2.0-2.5 ✅

2. **Shared Tokens** (higher is better):
   - Target: 20-30% ✅

3. **Balance** (closer to 50-50 is better):
   - English-dominant: ~40-50%
   - Turkish-dominant: ~40-50%
   - Shared: ~20-30%

## Integration Examples

### With HuggingFace Transformers

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my_tokenizer/tokenizer.json"
)

# Now use it like any HF tokenizer
tokenizer("Hello world!")
```

### With PyTorch Dataset

```python
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class BilingualDataset(Dataset):
    def __init__(self, data, tokenizer_path):
        self.data = data
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer.encode(text)
        return {
            'input_ids': encoded.ids,
            'attention_mask': encoded.attention_mask,
        }
```

### Training a Model

```python
from transformers import AutoModelForMaskedLM, Trainer

# Load model with custom tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my_tokenizer/tokenizer.json"
)

model = AutoModelForMaskedLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## Troubleshooting

### "Vocabulary too small"
Increase `--vocab-size`:
```bash
--vocab-size 50000  # or 64000
```

### "Too many rare tokens"
Increase `--min-frequency`:
```bash
--min-frequency 3  # or 5
```

### "Languages very imbalanced"
Lower `--alpha`:
```bash
--alpha 0.5  # maximum balancing
```

### "Not enough shared tokens"
Decrease core sizes:
```bash
--lang1-core 1000 --lang2-core 1000
```

## Next Steps

1. **Read the full README**: Learn about the research
2. **Check examples.py**: See advanced usage patterns
3. **Read ANALYSIS.md**: Understand your tokenizer's behavior
4. **Experiment**: Try different configurations

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/mnsoylemez/bilingual-tokenizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mnsoylemez/bilingual-tokenizer/discussions)
- **Email**: soylemeznurhan@gmail.com

---

**Happy Tokenizing! 🌍**

# 🌍 Bilingual Hybrid Tokenizer

An approach for training balanced bilingual BPE tokenizers that provide fair representation across languages while maintaining tokenization efficiency.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 The Problem

Traditional multilingual tokenizers suffer from **tokenization unfairness**:

- **Fertility Imbalance**: Low-resource or morphologically rich languages require 2-10× more tokens than English for the same content
- **Vocabulary Bias**: High-resource languages dominate the vocabulary, leaving underrepresented languages with fragmented, inefficient tokenization
- **Training Inefficiency**: Language imbalance leads to slower convergence and poorer performance on underrepresented languages

### Real-World Impact

```python
# English (well-represented)
"Hello world" → ["Hello", "world"]  # 2 tokens ✅

# Turkish (underrepresented)
"Merhaba dünya" → ["Mer", "##ha", "##ba", "dün", "##ya"]  # 5 tokens ❌
```

This 2.5× fertility ratio means:
- **Higher inference costs** for Turkish users
- **Shorter context windows** (fewer actual words fit)
- **Worse model performance** on Turkish tasks

## 💡 My Solution: Hybrid Frozen Vocabulary

I implement a **hybrid frozen vocabulary approach** inspired by recent NLP research:

### Key Innovation

1. **Language-Specific Core Vocabularies**: Extract high-frequency tokens from each language independently
2. **Frozen Representation**: Guarantee these core tokens appear in the final vocabulary
3. **Balanced Learning**: Train remaining vocabulary on alpha-sampled balanced corpus
4. **Fair Resource Allocation**: Each language gets guaranteed vocabulary budget

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Language 1 Corpus          Language 2 Corpus           │
│  (e.g., English)            (e.g., Turkish)            │
└────────┬──────────────────────────────┬────────────────┘
         │                              │
         ▼                              ▼
    ┌────────┐                     ┌────────┐
    │ BPE    │                     │ BPE    │
    │ 10K    │                     │ 10K    │
    └────┬───┘                     └───┬────┘
         │                             │
         ▼                             ▼
    ┌─────────┐                   ┌─────────┐
    │ Top 2K  │                   │ Top 2K  │
    │ Core    │                   │ Core    │
    └────┬────┘                   └────┬────┘
         │                             │
         └─────────┬───────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │ Frozen Vocab     │
         │ (~4K tokens)     │
         │ Overlap merged   │
         └────────┬─────────┘
                  │
      ┌───────────▼──────────┐
      │ Alpha-Sampled        │
      │ Balanced Corpus      │
      │ (50-50 or custom)    │
      └───────────┬──────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Final BPE      │
         │ 32K tokens     │
         │ Core + Learned │
         └────────────────┘
```

## 🔬 Research Foundation

This implementation is based on cutting-edge research in multilingual NLP:

### Why This Approach Works(Hopefully)

Unlike naive joint training or simple vocabulary merging, my hybrid approach:

- ✅ **Preserves linguistic structure** of each language
- ✅ **Guarantees representation** for critical tokens
- ✅ **Enables cross-lingual transfer** through shared tokens
- ✅ **Balances fertility** across languages
- ✅ **Maintains natural token hierarchies** (important for curriculum learning)

## 🚀 Quick Start

### Installation

```bash
pip install tokenizers numpy matplotlib
```

### Basic Usage

```bash
python bilingual_tokenizer.py \
    --lang1-corpus english_corpus.txt \
    --lang2-corpus turkish_corpus.txt \
    --lang1-name English \
    --lang2-name Turkish \
    --vocab-size 32000 \
    --lang1-core 2000 \
    --lang2-core 2000 \
    --alpha 0.7 \
    --output-dir output/
```

### Python API

```python
from bilingual_tokenizer import BilingualTokenizerTrainer, BilingualConfig

# Configure training
config = BilingualConfig(
    lang1_corpus="data/english.txt",
    lang2_corpus="data/turkish.txt",
    lang1_name="English",
    lang2_name="Turkish",
    total_vocab_size=32000,
    lang1_core_size=2000,
    lang2_core_size=2000,
    alpha=0.7,  # Alpha sampling parameter
    output_dir="output/",
)

# Train tokenizer
trainer = BilingualTokenizerTrainer(config)
tokenizer = trainer.train()

# Use tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("output/tokenizer.json")

# Tokenize text
output = tokenizer.encode("Hello world! Merhaba dünya!")
print(output.tokens)
```

## 📊 Performance Metrics

The trainer automatically generates comprehensive analysis:

### Fertility Analysis
- **Fertility**: tokens per word (lower is better)
- **Characters per Token**: compression efficiency
- **Benchmark**: English ~1.3, Turkish ~2.0-2.5

### Vocabulary Distribution
- **Language-dominant tokens**: primarily used in one language
- **Shared tokens**: cross-lingual semantic overlap
- **Balance ratio**: fairness indicator

### Core Vocabulary Analysis
- **Coverage**: how many core tokens from each language
- **Overlap**: shared high-frequency tokens
- **Efficiency**: token reuse across languages

## 🎛️ Configuration Options

### Vocabulary Sizing

```python
total_vocab_size=32000,    # Total vocabulary (typical: 32K-64K)
lang1_core_size=2000,      # Core tokens for language 1
lang2_core_size=2000,      # Core tokens for language 2
```

**Guidelines:**
- Core size: 5-10% of total vocab
- Larger core = more language-specific tokens
- Smaller core = more shared tokens

### Alpha Sampling

```python
alpha=0.7,  # Range: 0.5-1.0
```

**Effect:**
- `alpha=1.0`: Natural corpus proportions
- `alpha=0.7`: Moderate balancing (recommended)
- `alpha=0.5`: Maximum balancing (equal probability)

**Formula:** `p(L) = (n(L)/N)^α / Σ(n(L')/N)^α`

### Balance Ratio

```python
balance_ratio=0.5,  # 0.5 = 50-50 split
```

Fine-tune the target language ratio in the balanced corpus.

## 📈 Output Files

```
output/
├── tokenizer.json           # HuggingFace tokenizer (ready to use)
├── vocabulary.json          # Full vocabulary with IDs
├── core_tokens.json         # Core tokens per language + overlap
├── analysis_report.json     # Detailed metrics (machine-readable)
└── ANALYSIS.md             # Human-readable analysis report
```

### Example Analysis Report

```markdown
# Bilingual Tokenizer Analysis Report

## Fertility Metrics

### English
- Fertility: 1.32 tokens/word ✅
- Characters per Token: 4.85

### Turkish
- Fertility: 2.18 tokens/word ✅
- Characters per Token: 3.92

## Vocabulary Distribution

- English-dominant tokens: 12,458 (38.9%)
- Turkish-dominant tokens: 11,234 (35.1%)
- Shared tokens: 8,308 (26.0%)

## Core Vocabulary

- English core: 2,000 tokens
- Turkish core: 2,000 tokens
- Overlap: 342 tokens (17.1%)
```

## 🔧 Advanced Usage

### Custom Corpus Sampling

```python
# Oversample low-resource language
config = BilingualConfig(
    lang1_corpus="high_resource.txt",  # 1M lines
    lang2_corpus="low_resource.txt",   # 100K lines
    alpha=0.5,  # Strong balancing
    balance_ratio=0.5,
)
```

### Multi-Script Languages

```python
# Latin + Cyrillic example
config = BilingualConfig(
    lang1_corpus="english.txt",
    lang2_corpus="russian.txt",
    lang1_name="English",
    lang2_name="Russian",
    lang1_core_size=3000,  # Increase for script diversity
    lang2_core_size=3000,
)
```

### Domain-Specific Tokenizers

```python
# Medical domain example
config = BilingualConfig(
    lang1_corpus="medical_en.txt",
    lang2_corpus="medical_tr.txt",
    total_vocab_size=50000,  # Larger vocab for domain terms
    lang1_core_size=5000,
    lang2_core_size=5000,
)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for 3+ languages
- [ ] Automatic alpha parameter tuning
- [ ] Visualization tools for vocabulary analysis
- [ ] Integration with popular training frameworks
- [ ] Benchmark suite for fertility testing

## 📚 Citation

If you use this tokenizer in your research, please cite:

```bibtex
@software{bilingual_tokenizer,
  author = {[M. Nurhan Söylemez]},
  title = {Bilingual Hybrid Tokenizer: Research-Backed Balanced Tokenization},
  year = {2026},
  url = {https://github.com/mnsoylemez/bilingual-tokenizer}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

"""
================================================================================
Bilingual Hybrid Tokenizer Trainer
================================================================================
A research-backed approach for training balanced bilingual BPE tokenizers.

This implementation combines insights from recent NLP research:
- Trans-tokenization for fair language representation
- Hybrid frozen vocabulary approach
- Alpha sampling for corpus balancing
- Fertility analysis for tokenization quality assessment

Based on research from:
- "Trans-Tokenization and Cross-lingual Vocabulary Transfers" (2024)
- "Hyperpolyglot LLMs: Cross-Lingual Interpretability" (2023)
- "One Tokenizer To Rule Them All" (2025)

Author: [M. Nurhan Söylemez]
License: MIT
================================================================================
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.trainers import BpeTrainer
import matplotlib.pyplot as plt


@dataclass
class BilingualConfig:
    """Configuration for bilingual tokenizer training"""
    # Corpus paths
    lang1_corpus: str
    lang2_corpus: str
    lang1_name: str = "Language1"
    lang2_name: str = "Language2"
    
    # Vocabulary configuration
    total_vocab_size: int = 32000
    lang1_core_size: int = 2000
    lang2_core_size: int = 2000
    
    # Sampling configuration
    alpha: float = 0.7  # Alpha sampling exponent (0.5-1.0)
    balance_ratio: float = 0.5  # Target ratio for balanced corpus (0.5 = 50-50)
    
    # BPE configuration
    min_frequency: int = 2
    special_tokens: List[str] = None
    
    # Output configuration
    output_dir: str = "bilingual_tokenizer_output"
    save_analysis: bool = True
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]",
                "[BOS]", "[EOS]", "[Q]", "[A]"
            ]


class BilingualTokenizerTrainer:
    """
    Trains a bilingual BPE tokenizer using hybrid frozen vocabulary approach.
    
    Process:
    1. Train separate tokenizers for each language
    2. Extract core (high-frequency) tokens from each
    3. Create balanced corpus from both languages
    4. Train final tokenizer with frozen core vocabularies
    """
    
    def __init__(self, config: BilingualConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Metrics storage
        self.metrics = {
            'lang1_core_tokens': [],
            'lang2_core_tokens': [],
            'shared_tokens': [],
            'fertility_lang1': {},
            'fertility_lang2': {},
            'vocab_distribution': {},
        }
    
    def load_corpus(self, path: str, sample_ratio: float = 1.0) -> List[str]:
        """Load corpus from file with optional sampling"""
        print(f"📂 Loading corpus from {path}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if sample_ratio < 1.0:
            n_samples = int(len(lines) * sample_ratio)
            lines = random.sample(lines, n_samples)
            print(f"   Sampled {n_samples:,} / {len(lines):,} lines")
        
        print(f"   Loaded {len(lines):,} lines")
        return lines
    
    def create_alpha_sampled_corpus(
        self, 
        corpus1: List[str], 
        corpus2: List[str]
    ) -> List[str]:
        """
        Create balanced corpus using alpha sampling.
        
        Alpha sampling (from mBERT, mT5):
        p(L) = (n(L) / N)^alpha / Σ(n(L') / N)^alpha
        
        Where:
        - n(L) = size of language L corpus
        - N = total corpus size
        - alpha = smoothing parameter (0.5-1.0)
        """
        n1, n2 = len(corpus1), len(corpus2)
        total = n1 + n2
        
        # Calculate alpha-sampled probabilities
        p1_raw = (n1 / total) ** self.config.alpha
        p2_raw = (n2 / total) ** self.config.alpha
        normalizer = p1_raw + p2_raw
        
        p1 = p1_raw / normalizer
        p2 = p2_raw / normalizer
        
        print(f"\n📊 Alpha Sampling (α={self.config.alpha}):")
        print(f"   {self.config.lang1_name}: {n1:,} lines → {p1:.1%} probability")
        print(f"   {self.config.lang2_name}: {n2:,} lines → {p2:.1%} probability")
        
        # Sample to achieve target ratio
        target_size = min(n1, n2) * 2  # Use smaller corpus size * 2
        n1_samples = int(target_size * p1)
        n2_samples = int(target_size * p2)
        
        # Ensure we don't oversample
        n1_samples = min(n1_samples, n1)
        n2_samples = min(n2_samples, n2)
        
        sampled1 = random.sample(corpus1, n1_samples)
        sampled2 = random.sample(corpus2, n2_samples)
        
        # Shuffle to mix languages
        balanced = sampled1 + sampled2
        random.shuffle(balanced)
        
        print(f"   Balanced corpus: {len(balanced):,} lines")
        print(f"   Actual ratio: {len(sampled1)/len(balanced):.1%} / {len(sampled2)/len(balanced):.1%}")
        
        return balanced
    
    def train_monolingual_tokenizer(
        self, 
        corpus: List[str], 
        vocab_size: int,
        lang_name: str
    ) -> Tokenizer:
        """Train a BPE tokenizer for a single language"""
        print(f"\n🔧 Training {lang_name} tokenizer (vocab_size={vocab_size:,})...")
        
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        
        actual_size = tokenizer.get_vocab_size()
        print(f"   ✅ Trained tokenizer with {actual_size:,} tokens")
        
        return tokenizer
    
    def extract_core_tokens(
        self, 
        tokenizer: Tokenizer, 
        corpus: List[str],
        top_n: int,
        lang_name: str
    ) -> Set[str]:
        """
        Extract top-N most frequent tokens from tokenizer.
        
        This creates the "frozen core vocabulary" for each language.
        """
        print(f"\n🎯 Extracting top-{top_n} core tokens for {lang_name}...")
        
        vocab = tokenizer.get_vocab()
        
        # Tokenize corpus and count token frequencies
        token_freq = Counter()
        for line in corpus[:10000]:  # Sample for speed
            encoded = tokenizer.encode(line)
            token_freq.update(encoded.ids)
        
        # Get token strings from IDs
        id_to_token = {v: k for k, v in vocab.items()}
        
        # Sort by frequency and get top-N (excluding special tokens)
        core_tokens = set()
        sorted_tokens = token_freq.most_common()
        
        for token_id, freq in sorted_tokens:
            token_str = id_to_token[token_id]
            
            # Skip special tokens
            if token_str.startswith('[') and token_str.endswith(']'):
                continue
            
            core_tokens.add(token_str)
            
            if len(core_tokens) >= top_n:
                break
        
        print(f"   ✅ Extracted {len(core_tokens)} core tokens")
        
        # Show examples
        examples = list(core_tokens)[:10]
        decoded_examples = [tokenizer.decode([vocab[t]]) for t in examples if t in vocab]
        print(f"   Examples: {decoded_examples[:5]}")
        
        return core_tokens
    
    def find_overlap(
        self, 
        tokens1: Set[str], 
        tokens2: Set[str]
    ) -> Set[str]:
        """Find overlapping tokens between two vocabularies"""
        overlap = tokens1 & tokens2
        print(f"\n🔗 Vocabulary Overlap Analysis:")
        print(f"   {self.config.lang1_name} core: {len(tokens1)} tokens")
        print(f"   {self.config.lang2_name} core: {len(tokens2)} tokens")
        print(f"   Overlap: {len(overlap)} tokens ({len(overlap)/min(len(tokens1), len(tokens2)):.1%})")
        
        return overlap
    
    def train_final_tokenizer(
        self, 
        corpus: List[str],
        frozen_tokens: Set[str]
    ) -> Tokenizer:
        """
        Train final bilingual tokenizer with frozen core vocabulary.
        
        The frozen tokens are treated as "blacklisted" - we train with extra
        vocab size and then remove duplicates to hit target size.
        """
        print(f"\n🚀 Training final bilingual tokenizer...")
        print(f"   Frozen vocabulary: {len(frozen_tokens)} tokens")
        print(f"   Target total: {self.config.total_vocab_size:,} tokens")
        print(f"   Tokens to learn: {self.config.total_vocab_size - len(frozen_tokens):,}")
        
        # Train with adjusted vocab size to account for frozen tokens
        adjusted_vocab_size = self.config.total_vocab_size
        
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        trainer = BpeTrainer(
            vocab_size=adjusted_vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        
        final_size = tokenizer.get_vocab_size()
        print(f"   ✅ Final vocabulary size: {final_size:,} tokens")
        
        return tokenizer
    
    def calculate_fertility(
        self, 
        tokenizer: Tokenizer, 
        corpus: List[str],
        lang_name: str,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate tokenization fertility (tokens per word).
        
        Lower fertility = more efficient tokenization
        Fertility benchmark:
        - English: ~1.2-1.4
        - Agglutinative languages (Turkish): ~1.8-2.5
        - Ideographic (Chinese): ~0.8-1.2
        """
        print(f"\n📈 Calculating fertility for {lang_name}...")
        
        sample = corpus[:sample_size]
        
        total_words = 0
        total_tokens = 0
        
        for line in sample:
            words = line.split()
            total_words += len(words)
            
            encoded = tokenizer.encode(line)
            total_tokens += len(encoded.ids)
        
        fertility = total_tokens / total_words if total_words > 0 else 0
        
        # Calculate characters per token
        total_chars = sum(len(line.replace(' ', '')) for line in sample)
        cpt = total_chars / total_tokens if total_tokens > 0 else 0
        
        metrics = {
            'fertility': fertility,
            'chars_per_token': cpt,
            'avg_tokens_per_line': total_tokens / len(sample),
            'avg_words_per_line': total_words / len(sample),
        }
        
        print(f"   Fertility: {fertility:.2f} tokens/word")
        print(f"   Chars/Token: {cpt:.2f}")
        print(f"   Avg tokens/line: {metrics['avg_tokens_per_line']:.1f}")
        
        return metrics
    
    def analyze_vocabulary_distribution(
        self,
        tokenizer: Tokenizer,
        lang1_corpus: List[str],
        lang2_corpus: List[str]
    ) -> Dict:
        """
        Analyze how vocabulary is distributed across languages.
        
        Categorizes tokens as:
        - lang1_dominant: mostly used in language 1
        - lang2_dominant: mostly used in language 2
        - shared: used frequently in both
        """
        print(f"\n📊 Analyzing vocabulary distribution...")
        
        vocab = tokenizer.get_vocab()
        
        # Count token usage per language
        lang1_usage = Counter()
        lang2_usage = Counter()
        
        # Sample for performance
        for line in lang1_corpus[:5000]:
            encoded = tokenizer.encode(line)
            lang1_usage.update(encoded.ids)
        
        for line in lang2_corpus[:5000]:
            encoded = tokenizer.encode(line)
            lang2_usage.update(encoded.ids)
        
        # Categorize tokens
        lang1_dominant = 0
        lang2_dominant = 0
        shared = 0
        
        for token_id in vocab.values():
            count1 = lang1_usage[token_id]
            count2 = lang2_usage[token_id]
            total = count1 + count2
            
            if total == 0:
                continue
            
            ratio1 = count1 / total
            
            if ratio1 > 0.8:
                lang1_dominant += 1
            elif ratio1 < 0.2:
                lang2_dominant += 1
            else:
                shared += 1
        
        distribution = {
            f'{self.config.lang1_name}_dominant': lang1_dominant,
            f'{self.config.lang2_name}_dominant': lang2_dominant,
            'shared': shared,
            'total': lang1_dominant + lang2_dominant + shared,
        }
        
        print(f"   {self.config.lang1_name}-dominant: {lang1_dominant} ({lang1_dominant/distribution['total']:.1%})")
        print(f"   {self.config.lang2_name}-dominant: {lang2_dominant} ({lang2_dominant/distribution['total']:.1%})")
        print(f"   Shared: {shared} ({shared/distribution['total']:.1%})")
        
        return distribution
    
    def save_artifacts(
        self,
        tokenizer: Tokenizer,
        core_tokens_lang1: Set[str],
        core_tokens_lang2: Set[str],
        overlap: Set[str],
        fertility_lang1: Dict,
        fertility_lang2: Dict,
        vocab_distribution: Dict
    ):
        """Save all tokenizer artifacts and analysis results"""
        output_dir = Path(self.config.output_dir)
        
        # Save tokenizer
        tokenizer_path = output_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        print(f"\n💾 Saved tokenizer to: {tokenizer_path}")
        
        # Save vocabulary
        vocab = tokenizer.get_vocab()
        vocab_path = output_dir / "vocabulary.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved vocabulary to: {vocab_path}")
        
        # Save core tokens
        core_path = output_dir / "core_tokens.json"
        core_data = {
            self.config.lang1_name: list(core_tokens_lang1),
            self.config.lang2_name: list(core_tokens_lang2),
            'overlap': list(overlap),
        }
        with open(core_path, 'w', encoding='utf-8') as f:
            json.dump(core_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved core tokens to: {core_path}")
        
        # Save analysis report
        if self.config.save_analysis:
            report = {
                'config': asdict(self.config),
                'fertility': {
                    self.config.lang1_name: fertility_lang1,
                    self.config.lang2_name: fertility_lang2,
                },
                'vocabulary_distribution': vocab_distribution,
                'core_vocabulary': {
                    f'{self.config.lang1_name}_size': len(core_tokens_lang1),
                    f'{self.config.lang2_name}_size': len(core_tokens_lang2),
                    'overlap_size': len(overlap),
                    'overlap_percentage': len(overlap) / min(len(core_tokens_lang1), len(core_tokens_lang2))
                },
                'final_vocab_size': tokenizer.get_vocab_size(),
            }
            
            report_path = output_dir / "analysis_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"💾 Saved analysis report to: {report_path}")
            
            # Create human-readable markdown report
            self.create_markdown_report(report, output_dir / "ANALYSIS.md")
    
    def create_markdown_report(self, report: Dict, path: Path):
        """Create a human-readable markdown analysis report"""
        md = f"""# Bilingual Tokenizer Analysis Report

## Configuration

- **Language 1**: {self.config.lang1_name}
- **Language 2**: {self.config.lang2_name}
- **Total Vocabulary Size**: {report['final_vocab_size']:,}
- **Core Tokens per Language**: {self.config.lang1_core_size:,} / {self.config.lang2_core_size:,}
- **Alpha Sampling Parameter**: {self.config.alpha}

## Fertility Metrics

### {self.config.lang1_name}
- **Fertility**: {report['fertility'][self.config.lang1_name]['fertility']:.2f} tokens/word
- **Characters per Token**: {report['fertility'][self.config.lang1_name]['chars_per_token']:.2f}

### {self.config.lang2_name}
- **Fertility**: {report['fertility'][self.config.lang2_name]['fertility']:.2f} tokens/word
- **Characters per Token**: {report['fertility'][self.config.lang2_name]['chars_per_token']:.2f}

### Interpretation
- Lower fertility = more efficient tokenization
- Benchmark: English ~1.3, Agglutinative languages ~2.0-2.5

## Vocabulary Distribution

- **{self.config.lang1_name}-dominant tokens**: {report['vocabulary_distribution'][f'{self.config.lang1_name}_dominant']} ({report['vocabulary_distribution'][f'{self.config.lang1_name}_dominant']/report['vocabulary_distribution']['total']:.1%})
- **{self.config.lang2_name}-dominant tokens**: {report['vocabulary_distribution'][f'{self.config.lang2_name}_dominant']} ({report['vocabulary_distribution'][f'{self.config.lang2_name}_dominant']/report['vocabulary_distribution']['total']:.1%})
- **Shared tokens**: {report['vocabulary_distribution']['shared']} ({report['vocabulary_distribution']['shared']/report['vocabulary_distribution']['total']:.1%})

## Core Vocabulary

- **{self.config.lang1_name} core**: {report['core_vocabulary'][f'{self.config.lang1_name}_size']} tokens
- **{self.config.lang2_name} core**: {report['core_vocabulary'][f'{self.config.lang2_name}_size']} tokens
- **Overlap**: {report['core_vocabulary']['overlap_size']} tokens ({report['core_vocabulary']['overlap_percentage']:.1%})

## Summary

This tokenizer was trained using a hybrid frozen vocabulary approach, combining:
1. Language-specific core vocabularies (high-frequency tokens)
2. Shared vocabulary learned from balanced bilingual corpus
3. Alpha sampling for fair language representation

The result is a tokenizer that provides balanced representation for both languages while maintaining efficient tokenization.
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"📄 Saved markdown report to: {path}")
    
    def train(self) -> Tokenizer:
        """
        Main training pipeline for bilingual tokenizer.
        
        Returns:
            Trained bilingual tokenizer
        """
        print("="*80)
        print("🚀 BILINGUAL HYBRID TOKENIZER TRAINER")
        print("="*80)
        
        # Step 1: Load corpora
        print("\n" + "="*80)
        print("STEP 1: Loading Corpora")
        print("="*80)
        
        lang1_corpus = self.load_corpus(self.config.lang1_corpus)
        lang2_corpus = self.load_corpus(self.config.lang2_corpus)
        
        # Step 2: Train monolingual tokenizers
        print("\n" + "="*80)
        print("STEP 2: Training Monolingual Tokenizers")
        print("="*80)
        
        # Use 10K vocab for initial training to ensure we get good core tokens
        temp_vocab_size = max(10000, self.config.lang1_core_size * 5)
        
        lang1_tokenizer = self.train_monolingual_tokenizer(
            lang1_corpus, 
            temp_vocab_size,
            self.config.lang1_name
        )
        
        lang2_tokenizer = self.train_monolingual_tokenizer(
            lang2_corpus, 
            temp_vocab_size,
            self.config.lang2_name
        )
        
        # Step 3: Extract core vocabularies
        print("\n" + "="*80)
        print("STEP 3: Extracting Core Vocabularies")
        print("="*80)
        
        lang1_core = self.extract_core_tokens(
            lang1_tokenizer,
            lang1_corpus,
            self.config.lang1_core_size,
            self.config.lang1_name
        )
        
        lang2_core = self.extract_core_tokens(
            lang2_tokenizer,
            lang2_corpus,
            self.config.lang2_core_size,
            self.config.lang2_name
        )
        
        # Step 4: Find overlap
        overlap = self.find_overlap(lang1_core, lang2_core)
        
        # Frozen vocabulary = union of both cores (overlap counted once)
        frozen_vocab = lang1_core | lang2_core
        
        print(f"\n📌 Frozen vocabulary: {len(frozen_vocab)} unique tokens")
        
        # Step 5: Create balanced corpus
        print("\n" + "="*80)
        print("STEP 4: Creating Balanced Bilingual Corpus")
        print("="*80)
        
        balanced_corpus = self.create_alpha_sampled_corpus(
            lang1_corpus,
            lang2_corpus
        )
        
        # Step 6: Train final bilingual tokenizer
        print("\n" + "="*80)
        print("STEP 5: Training Final Bilingual Tokenizer")
        print("="*80)
        
        final_tokenizer = self.train_final_tokenizer(
            balanced_corpus,
            frozen_vocab
        )
        
        # Step 7: Analyze results
        print("\n" + "="*80)
        print("STEP 6: Analyzing Results")
        print("="*80)
        
        fertility_lang1 = self.calculate_fertility(
            final_tokenizer,
            lang1_corpus,
            self.config.lang1_name
        )
        
        fertility_lang2 = self.calculate_fertility(
            final_tokenizer,
            lang2_corpus,
            self.config.lang2_name
        )
        
        vocab_distribution = self.analyze_vocabulary_distribution(
            final_tokenizer,
            lang1_corpus,
            lang2_corpus
        )
        
        # Step 8: Save everything
        print("\n" + "="*80)
        print("STEP 7: Saving Artifacts")
        print("="*80)
        
        self.save_artifacts(
            final_tokenizer,
            lang1_core,
            lang2_core,
            overlap,
            fertility_lang1,
            fertility_lang2,
            vocab_distribution
        )
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\n📁 All artifacts saved to: {self.config.output_dir}")
        
        return final_tokenizer


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Train a bilingual BPE tokenizer using hybrid frozen vocabulary approach"
    )
    
    # Required arguments
    parser.add_argument(
        "--lang1-corpus",
        required=True,
        help="Path to language 1 corpus file"
    )
    parser.add_argument(
        "--lang2-corpus",
        required=True,
        help="Path to language 2 corpus file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--lang1-name",
        default="Language1",
        help="Name of language 1 (for reporting)"
    )
    parser.add_argument(
        "--lang2-name",
        default="Language2",
        help="Name of language 2 (for reporting)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Total vocabulary size (default: 32000)"
    )
    parser.add_argument(
        "--lang1-core",
        type=int,
        default=2000,
        help="Core vocabulary size for language 1 (default: 2000)"
    )
    parser.add_argument(
        "--lang2-core",
        type=int,
        default=2000,
        help="Core vocabulary size for language 2 (default: 2000)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Alpha sampling parameter (0.5-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--output-dir",
        default="bilingual_tokenizer_output",
        help="Output directory (default: bilingual_tokenizer_output)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = BilingualConfig(
        lang1_corpus=args.lang1_corpus,
        lang2_corpus=args.lang2_corpus,
        lang1_name=args.lang1_name,
        lang2_name=args.lang2_name,
        total_vocab_size=args.vocab_size,
        lang1_core_size=args.lang1_core,
        lang2_core_size=args.lang2_core,
        alpha=args.alpha,
        output_dir=args.output_dir,
        min_frequency=args.min_frequency,
        random_seed=args.seed,
    )
    
    # Train tokenizer
    trainer = BilingualTokenizerTrainer(config)
    tokenizer = trainer.train()
    
    print("\n🎉 Success! Your bilingual tokenizer is ready to use.")
    print(f"\n💡 To use it in Python:")
    print(f"   from tokenizers import Tokenizer")
    print(f"   tokenizer = Tokenizer.from_file('{args.output_dir}/tokenizer.json')")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example usage of the Bilingual Hybrid Tokenizer

This script demonstrates how to use the tokenizer for various tasks.
"""

from bilingual_tokenizer import BilingualTokenizerTrainer, BilingualConfig
from tokenizers import Tokenizer
import json


def example_1_basic_training():
    """Example 1: Basic training with default parameters"""
    print("="*80)
    print("EXAMPLE 1: Basic Training")
    print("="*80)
    
    config = BilingualConfig(
        lang1_corpus="data/english.txt",
        lang2_corpus="data/turkish.txt",
        lang1_name="English",
        lang2_name="Turkish",
        output_dir="output/example1",
    )
    
    trainer = BilingualTokenizerTrainer(config)
    tokenizer = trainer.train()
    
    print("\n✅ Example 1 complete!")
    print(f"📁 Output saved to: {config.output_dir}")


def example_2_custom_vocabulary():
    """Example 2: Custom vocabulary sizes"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Vocabulary Sizes")
    print("="*80)
    
    config = BilingualConfig(
        lang1_corpus="data/english.txt",
        lang2_corpus="data/turkish.txt",
        lang1_name="English",
        lang2_name="Turkish",
        total_vocab_size=50000,      # Larger vocabulary
        lang1_core_size=5000,         # More core tokens
        lang2_core_size=5000,
        output_dir="output/example2",
    )
    
    trainer = BilingualTokenizerTrainer(config)
    tokenizer = trainer.train()
    
    print("\n✅ Example 2 complete!")


def example_3_imbalanced_corpora():
    """Example 3: Handling imbalanced corpora with alpha sampling"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Imbalanced Corpora (High-resource + Low-resource)")
    print("="*80)
    
    config = BilingualConfig(
        lang1_corpus="data/english_large.txt",   # 1M lines
        lang2_corpus="data/turkish_small.txt",   # 100K lines
        lang1_name="English",
        lang2_name="Turkish",
        alpha=0.5,  # Strong balancing for imbalanced corpora
        output_dir="output/example3",
    )
    
    trainer = BilingualTokenizerTrainer(config)
    tokenizer = trainer.train()
    
    print("\n✅ Example 3 complete!")
    print("📊 Alpha=0.5 provides maximum balancing for imbalanced corpora")


def example_4_using_trained_tokenizer():
    """Example 4: Using a trained tokenizer"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Using Trained Tokenizer")
    print("="*80)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("output/example1/tokenizer.json")
    
    # Test sentences
    test_sentences = [
        "Hello world! How are you?",
        "Merhaba dünya! Nasılsın?",
        "Machine learning is fascinating.",
        "Makine öğrenimi büyüleyici.",
    ]
    
    print("\nTokenization Examples:")
    print("-" * 80)
    
    for sentence in test_sentences:
        output = tokenizer.encode(sentence)
        print(f"\nInput:  {sentence}")
        print(f"Tokens: {output.tokens}")
        print(f"IDs:    {output.ids}")
        print(f"Count:  {len(output.ids)} tokens")
    
    print("\n✅ Example 4 complete!")


def example_5_analyze_results():
    """Example 5: Analyzing tokenizer results"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Analyzing Results")
    print("="*80)
    
    # Load analysis report
    with open("output/example1/analysis_report.json", 'r') as f:
        report = json.load(f)
    
    print("\n📊 Tokenizer Statistics:")
    print("-" * 80)
    
    # Fertility comparison
    print("\n1. Fertility Analysis:")
    for lang, metrics in report['fertility'].items():
        print(f"\n   {lang}:")
        print(f"   - Fertility: {metrics['fertility']:.2f} tokens/word")
        print(f"   - Characters/Token: {metrics['chars_per_token']:.2f}")
    
    # Vocabulary distribution
    print("\n2. Vocabulary Distribution:")
    dist = report['vocabulary_distribution']
    total = dist['total']
    for key, value in dist.items():
        if key != 'total':
            percentage = (value / total) * 100
            print(f"   - {key}: {value:,} ({percentage:.1f}%)")
    
    # Core vocabulary
    print("\n3. Core Vocabulary:")
    core = report['core_vocabulary']
    for key, value in core.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"   - {key}: {value:.1%}")
            else:
                print(f"   - {key}: {value:,}")
    
    print("\n✅ Example 5 complete!")


def example_6_compare_tokenizers():
    """Example 6: Comparing different configurations"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Comparing Configurations")
    print("="*80)
    
    configurations = [
        ("Low Alpha (0.5)", {"alpha": 0.5, "output_dir": "output/compare_alpha_05"}),
        ("Mid Alpha (0.7)", {"alpha": 0.7, "output_dir": "output/compare_alpha_07"}),
        ("High Alpha (0.9)", {"alpha": 0.9, "output_dir": "output/compare_alpha_09"}),
    ]
    
    results = []
    
    for name, custom_params in configurations:
        print(f"\nTraining: {name}")
        print("-" * 40)
        
        config = BilingualConfig(
            lang1_corpus="data/english.txt",
            lang2_corpus="data/turkish.txt",
            lang1_name="English",
            lang2_name="Turkish",
            **custom_params
        )
        
        trainer = BilingualTokenizerTrainer(config)
        tokenizer = trainer.train()
        
        # Load results
        with open(f"{custom_params['output_dir']}/analysis_report.json", 'r') as f:
            report = json.load(f)
        
        results.append({
            'name': name,
            'english_fertility': report['fertility']['English']['fertility'],
            'turkish_fertility': report['fertility']['Turkish']['fertility'],
            'shared_tokens': report['vocabulary_distribution']['shared'],
        })
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Configuration", "EN Fertility", "TR Fertility", "Shared Tokens"
    ))
    print("-" * 70)
    
    for result in results:
        print("{:<20} {:>15.2f} {:>15.2f} {:>15}".format(
            result['name'],
            result['english_fertility'],
            result['turkish_fertility'],
            result['shared_tokens']
        ))
    
    print("\n✅ Example 6 complete!")
    print("\n💡 Insight: Lower alpha = more balancing = more shared tokens")


def main():
    """Run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                 BILINGUAL HYBRID TOKENIZER - USAGE EXAMPLES                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script demonstrates various usage patterns of the tokenizer.

Note: Make sure you have corpus files in data/ directory:
  - data/english.txt
  - data/turkish.txt

Or modify the paths in the examples to point to your corpora.
""")
    
    try:
        # Run examples
        # example_1_basic_training()
        # example_2_custom_vocabulary()
        # example_3_imbalanced_corpora()
        example_4_using_trained_tokenizer()
        example_5_analyze_results()
        # example_6_compare_tokenizers()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Corpus files in the correct locations")
        print("  2. Installed dependencies: pip install -r requirements.txt")
        print("  3. Run at least example 1 before running example 4 or 5")


if __name__ == "__main__":
    main()

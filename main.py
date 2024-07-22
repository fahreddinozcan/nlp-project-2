import os
import random
import codecs
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Task 1: Data Exploration

def read_file(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as file:
        return file.readlines()

def analyze_data(en_sentences, fr_sentences):
    en_lengths = [len(sent.split()) for sent in en_sentences]
    fr_lengths = [len(sent.split()) for sent in fr_sentences]
    
    print(f"Total number of sentences: {len(en_sentences)}")
    print(f"Average English sentence length: {sum(en_lengths) / len(en_lengths):.2f} words")
    print(f"Average French sentence length: {sum(fr_lengths) / len(fr_lengths):.2f} words")
    
    plt.figure(figsize=(10, 6))
    plt.hist(en_lengths, bins=50, alpha=0.5, label='English')
    plt.hist(fr_lengths, bins=50, alpha=0.5, label='French')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Lengths')
    plt.legend()
    plt.savefig('sentence_length_distribution.png')
    plt.close()
    
    en_words = [word for sent in en_sentences for word in sent.split()]
    fr_words = [word for sent in fr_sentences for word in sent.split()]
    
    print(f"Total English words: {len(en_words)}")
    print(f"Total French words: {len(fr_words)}")
    print(f"Unique English words: {len(set(en_words))}")
    print(f"Unique French words: {len(set(fr_words))}")
    
    en_common = Counter(en_words).most_common(10)
    fr_common = Counter(fr_words).most_common(10)
    
    print("Top 10 English words:", en_common)
    print("Top 10 French words:", fr_common)

def sample_data(en_sentences, fr_sentences, sample_ratio=0.1):
    sample_size = int(len(en_sentences) * sample_ratio)
    indices = random.sample(range(len(en_sentences)), sample_size)
    return [en_sentences[i] for i in indices], [fr_sentences[i] for i in indices]

# Task 2: Pre-processing

def preprocess(sentences):
    processed = []
    for sent in sentences:
        # Lowercase the text
        sent = sent.lower()
        # Remove lines with XML tags
        if not sent.startswith('<'):
            # Strip whitespace
            sent = sent.strip()
            if sent:  # Ensure the sentence is not empty
                processed.append(sent)
    return processed

# Main execution

if __name__ == "__main__":
    en_file = './fr-en/europarl-v7.fr-en.en'  # Update with your file path
    fr_file = './fr-en/europarl-v7.fr-en.fr'  # Update with your file path
    
    print("Reading files...")
    en_sentences = read_file(en_file)
    fr_sentences = read_file(fr_file)
    
    print("Analyzing data...")
    analyze_data(en_sentences, fr_sentences)
    
    print("Sampling data...")
    en_sample, fr_sample = sample_data(en_sentences, fr_sentences)
    
    print("Pre-processing data...")
    en_processed = preprocess(en_sample)
    fr_processed = preprocess(fr_sample)
    
    print(f"Original sample size: {len(en_sample)}")
    print(f"Processed sample size: {len(en_processed)}")
    
    # Save processed data
    with codecs.open('en_processed.txt', 'w', 'utf-8') as f:
        f.write('\n'.join(en_processed))
    with codecs.open('fr_processed.txt', 'w', 'utf-8') as f:
        f.write('\n'.join(fr_processed))
    
    print("Processed data saved to en_processed.txt and fr_processed.txt")
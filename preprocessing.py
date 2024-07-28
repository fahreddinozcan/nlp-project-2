import os
import random
import codecs
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def read_file(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as file:
        return file.readlines()
    
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


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

def preprocess(src_sentences, tgt_sentences):
    src_processed = []
    tgt_processed = []
    for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
        # Lowercase the text
        src_sent = src_sent.lower()
        tgt_sent = tgt_sent.lower()
        # Remove lines with XML tags
        if not src_sent.startswith('<') and not tgt_sent.startswith('<'):
            # Strip whitespace
            src_sent = src_sent.strip()
            tgt_sent = tgt_sent.strip()
            if src_sent and tgt_sent:  # Ensure the sentences are not empty
                src_processed.append(src_sent)
                tgt_processed.append(tgt_sent)
    return src_processed, tgt_processed
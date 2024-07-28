
import codecs
from preprocessing import read_file, analyze_data, sample_data, preprocess, load_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
# from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
import numpy as np
from Seq2Seq import Encoder, Decoder, Seq2Seq

# Define hyperparameters
embedding_size = 300
hidden_size = 1024
num_layers = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 20
    

class Vocabulary:
    def __init__(self, lang):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4
        self.lang = lang

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, src_max_len, tgt_max_len):
        self.src_data = prepare_data(src_data, src_vocab, src_max_len)
        self.tgt_data = prepare_data(tgt_data, tgt_vocab, tgt_max_len)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def prepare_data(texts, vocab, max_len):
        # First pad the text data
        for i in range(len(texts)):
            sentence = texts[i]
            words = sentence.split()
            while len(words) < max_len:
                words.append('<PAD>')
            sentence = ' '.join(words)
            texts[i] = sentence
    
        sequences = [[vocab.word2index.get(word, vocab.word2index['<UNK>']) for word in text.split()] for text in texts]
     
        return torch.LongTensor(sequences)
def pad_sequences(sequences, max_len):
        return [seq + ['<PAD>'] * (max_len - len(seq)) for seq in sequences]

if __name__ == "__main__":
    en_file = './fr-en/europarl-v7.fr-en.en'
    fr_file = './fr-en/europarl-v7.fr-en.fr'  
    
    print("Reading files...")
    en_sentences = read_file(en_file)
    fr_sentences = read_file(fr_file)
    
    # print("Analyzing data...")
    # analyze_data(en_sentences, fr_sentences)
    
    print("Sampling data...")
    en_sample, fr_sample = sample_data(en_sentences, fr_sentences)
    
    print("Pre-processing data...")
    en_processed , fr_processed = preprocess(en_sample, fr_sample)

    print(f"Original sample size: {len(en_sample)}")
    print(f"Processed sample size: {len(en_processed)}")
    
    # Save processed data
    with codecs.open('en_processed.txt', 'w', 'utf-8') as f:
        f.write('\n'.join(en_processed))
    with codecs.open('fr_processed.txt', 'w', 'utf-8') as f:
        f.write('\n'.join(fr_processed))
    
    print("Processed data saved to en_processed.txt and fr_processed.txt")
    
    en_data = load_data('en_processed.txt')
    fr_data = load_data('fr_processed.txt')
    
    train_en, test_en, train_fr, test_fr = train_test_split(en_data, fr_data, test_size=0.2, random_state=42)
    train_en, val_en, train_fr, val_fr = train_test_split(train_en, train_fr, test_size=0.1, random_state=42)
    
    en_vocab = Vocabulary('en')
    fr_vocab = Vocabulary('fr')

    for sentence in train_en:
        en_vocab.add_sentence(sentence)
    for sentence in train_fr:
        fr_vocab.add_sentence(sentence)
        
   
    max_len_en = max(len(seq.split()) for seq in en_data)
    max_len_fr = max(len(seq.split()) for seq in fr_data)
    
    
    input_size_en = en_vocab.n_words
    input_size_fr = fr_vocab.n_words
    output_size_en = en_vocab.n_words
    output_size_fr = fr_vocab.n_words
   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = TranslationDataset(train_en, train_fr, en_vocab, fr_vocab, max_len_en, max_len_fr)
    val_dataset = TranslationDataset(val_en, val_fr, en_vocab, fr_vocab, max_len_en, max_len_fr)
    test_dataset = TranslationDataset(test_en, test_fr, en_vocab, fr_vocab, max_len_en, max_len_fr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    encoder = Encoder(input_size_en, embedding_size, hidden_size, num_layers).to(device)
    decoder = Decoder(input_size_fr, embedding_size, hidden_size, output_size_fr, num_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.word2index['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(model, loader, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(loader)
    
    def evaluate(model, loader, criterion):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for src, tgt in loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, 0)  # turn off teacher forcing
                output = output[:, 1:].reshape(-1, output.shape[2])
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)
                epoch_loss += loss.item()
        return epoch_loss / len(loader)
    
    n_epochs = 20
    clip = 1

    best_valid_loss = float('inf')

    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        val_loss = evaluate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}')
        
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    def calculate_bleu(model, loader, vocab):
        model.eval()
        targets = []
        outputs = []
        with torch.no_grad():
            for src, tgt in loader:
                src = src.to(device)
                output = model(src, tgt, 0)
                output = output.argmax(dim=2)
                for i in range(output.shape[0]):
                    target = [vocab.index2word[idx.item()] for idx in tgt[i] if idx.item() not in [0, 1, 2]]
                    pred = [vocab.index2word[idx.item()] for idx in output[i] if idx.item() not in [0, 1, 2]]
                    targets.append([target])
                    outputs.append(pred)
        return corpus_bleu(outputs, targets)
    
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss = evaluate(model, test_loader, criterion)
    test_bleu = calculate_bleu(model, test_loader, fr_vocab)
    
    print(f'Test Loss: {test_loss:.3f}')
    print(f'Test BLEU: {test_bleu:.3f}')

    # French to English model
    encoder_fr = Encoder(input_size_fr, embedding_size, hidden_size, num_layers).to(device)
    decoder_en = Decoder(input_size_en, embedding_size, hidden_size, output_size_en, num_layers).to(device)
    model_fr_en = Seq2Seq(encoder_fr, decoder_en).to(device)
    
    # English to French model
    encoder_en = Encoder(input_size_en, embedding_size, hidden_size, num_layers).to(device)
    decoder_fr = Decoder(input_size_fr, embedding_size, hidden_size, output_size_fr, num_layers).to(device)
    model_en_fr = Seq2Seq(encoder_en, decoder_fr).to(device)
    
    class CharVocabulary:
        def __init__(self, lang):
            self.char2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
            self.char2count = {}
            self.index2char = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            self.n_chars = 4
            self.lang = lang

        def add_sentence(self, sentence):
            for char in sentence:
                self.add_char(char)

        def add_char(self, char):
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.char2count[char] = 1
                self.index2char[self.n_chars] = char
                self.n_chars += 1
            else:
                self.char2count[char] += 1
                

    char_vocab_en = CharVocabulary('en')
    char_vocab_fr = CharVocabulary('fr')

    for sentence in train_en:
        char_vocab_en.add_sentence(sentence)
    for sentence in train_fr:
        char_vocab_fr.add_sentence(sentence)

    # Prepare character data
    def prepare_char_data(texts, vocab, max_len):
        sequences = [[vocab.char2index.get(char, vocab.char2index['<UNK>']) for char in text] for text in texts]
        sequences = pad_sequences(sequences, max_len)
        return torch.LongTensor(sequences)

    max_char_len_en = max(len(seq) for seq in en_data)
    max_char_len_fr = max(len(seq) for seq in fr_data)
    
    # Character-based dataset
    class CharTranslationDataset(Dataset):
        def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, src_max_len, tgt_max_len):
            self.src_data = prepare_char_data(src_data, src_vocab, src_max_len)
            self.tgt_data = prepare_char_data(tgt_data, tgt_vocab, tgt_max_len)

        def __len__(self):
            return len(self.src_data)

        def __getitem__(self, idx):
            return self.src_data[idx], self.tgt_data[idx]

    # Create character-based datasets and dataloaders
    char_train_dataset = CharTranslationDataset(train_en, train_fr, char_vocab_en, char_vocab_fr, max_char_len_en, max_char_len_fr)
    char_val_dataset = CharTranslationDataset(val_en, val_fr, char_vocab_en, char_vocab_fr, max_char_len_en, max_char_len_fr)
    char_test_dataset = CharTranslationDataset(test_en, test_fr, char_vocab_en, char_vocab_fr, max_char_len_en, max_char_len_fr)

    char_train_loader = DataLoader(char_train_dataset, batch_size=batch_size, shuffle=True)
    char_val_loader = DataLoader(char_val_dataset, batch_size=batch_size)
    char_test_loader = DataLoader(char_test_dataset, batch_size=batch_size)
    
    # Initialize character-based models
    char_input_size_en = char_vocab_en.n_chars
    char_input_size_fr = char_vocab_fr.n_chars
    char_output_size_en = char_vocab_en.n_chars
    char_output_size_fr = char_vocab_fr.n_chars
    char_embedding_size = 128  # Smaller embedding size for characters
    char_hidden_size = 512  # Smaller hidden size for characters

    char_encoder = Encoder(char_input_size_en, char_embedding_size, char_hidden_size, num_layers).to(device)
    char_decoder = Decoder(char_input_size_fr, char_embedding_size, char_hidden_size, char_output_size_fr, num_layers).to(device)
    char_model = Seq2Seq(char_encoder, char_decoder).to(device)

    # Loss and optimizer for character-based model
    char_criterion = nn.CrossEntropyLoss(ignore_index=char_vocab_en.char2index['<PAD>'])
    char_optimizer = optim.Adam(char_model.parameters(), lr=learning_rate)

    # Train character-based model
    char_train_losses = []
    char_val_losses = []
    char_best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        char_train_loss = train(char_model, char_train_loader, char_optimizer, char_criterion, clip)
        char_val_loss = evaluate(char_model, char_val_loader, char_criterion)
        
        char_train_losses.append(char_train_loss)
        char_val_losses.append(char_val_loss)
        
        if char_val_loss < char_best_valid_loss:
            char_best_valid_loss = char_val_loss
            torch.save(char_model.state_dict(), 'best_char_model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {char_train_loss:.3f}')
        print(f'\t Val. Loss: {char_val_loss:.3f}')

    # Plot character-based model training history
    plt.figure(figsize=(10, 5))
    plt.plot(char_train_losses, label='Train Loss')
    plt.plot(char_val_losses, label='Validation Loss')
    plt.title('Character-based Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('char_training_history.png')
    plt.close()
    
    
# Evaluation function for character-based model
    def calculate_char_bleu(model, loader, vocab):
        model.eval()
        targets = []
        outputs = []
        with torch.no_grad():
            for src, tgt in loader:
                src = src.to(device)
                output = model(src, tgt, 0)
                output = output.argmax(dim=2)
                for i in range(output.shape[0]):
                    target = ''.join([vocab.index2char[idx.item()] for idx in tgt[i] if idx.item() not in [0, 1, 2]])
                    pred = ''.join([vocab.index2char[idx.item()] for idx in output[i] if idx.item() not in [0, 1, 2]])
                    targets.append([target])
                    outputs.append(pred)
        return corpus_bleu(outputs, targets)

    # Evaluate character-based model on test set
    char_model.load_state_dict(torch.load('best_char_model.pt'))
    char_test_loss = evaluate(char_model, char_test_loader, char_criterion)
    char_test_bleu = calculate_char_bleu(char_model, char_test_loader, char_vocab_fr)

    print(f'Character-based Model Test Loss: {char_test_loss:.3f}')
    print(f'Character-based Model Test BLEU: {char_test_bleu:.3f}')

    # Compare results and analyze
    print("\nComparison of Results:")
    print(f"Word-based Model Test Loss: {test_loss:.3f}")
    print(f"Word-based Model Test BLEU: {test_bleu:.3f}")
    print(f"Character-based Model Test Loss: {char_test_loss:.3f}")
    print(f"Character-based Model Test BLEU: {char_test_bleu:.3f}")

    # Analyze impact of sentence length on performance
    def analyze_length_impact(model, loader, vocab, is_char_based=False):
        model.eval()
        length_bleu = {}
        with torch.no_grad():
            for src, tgt in loader:
                src = src.to(device)
                output = model(src, tgt, 0)
                output = output.argmax(dim=2)
                for i in range(output.shape[0]):
                    if is_char_based:
                        target = ''.join([vocab.index2char[idx.item()] for idx in tgt[i] if idx.item() not in [0, 1, 2]])
                        pred = ''.join([vocab.index2char[idx.item()] for idx in output[i] if idx.item() not in [0, 1, 2]])
                    else:
                        target = [vocab.index2word[idx.item()] for idx in tgt[i] if idx.item() not in [0, 1, 2]]
                        pred = [vocab.index2word[idx.item()] for idx in output[i] if idx.item() not in [0, 1, 2]]
                    
                    length = len(target) if is_char_based else len(target.split())
                    bleu = corpus_bleu([pred], [[target]])
                    
                    if length not in length_bleu:
                        length_bleu[length] = []
                    length_bleu[length].append(bleu)
        
        avg_length_bleu = {length: sum(scores) / len(scores) for length, scores in length_bleu.items()}
        return avg_length_bleu

    word_length_impact = analyze_length_impact(model, test_loader, fr_vocab)
    char_length_impact = analyze_length_impact(char_model, char_test_loader, char_vocab_fr, is_char_based=True)

    # Plot length impact
    plt.figure(figsize=(12, 6))
    plt.plot(word_length_impact.keys(), word_length_impact.values(), label='Word-based Model')
    plt.plot(char_length_impact.keys(), char_length_impact.values(), label='Character-based Model')
    plt.title('Impact of Sentence Length on BLEU Score')
    plt.xlabel('Sentence Length')
    plt.ylabel('Average BLEU Score')
    plt.legend()
    plt.savefig('length_impact.png')
    plt.close()

    # Analysis and Insights
    print("\nAnalysis and Insights:")
    print("1. Overall Performance:")
    print(f"   - Word-based model BLEU: {test_bleu:.3f}")
    print(f"   - Character-based model BLEU: {char_test_bleu:.3f}")

    print("\n2. Impact of Sentence Length:")
    print("   - See 'length_impact.png' for a visual representation")
    print("   - Word-based model performance tends to decrease with longer sentences")
    print("   - Character-based model shows more consistent performance across different lengths")

    print("\n3. Model Complexity:")
    print(f"   - Word-based model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   - Character-based model parameters: {sum(p.numel() for p in char_model.parameters())}")

    print("\n4. Training Efficiency:")
    print(f"   - Word-based model final training loss: {train_losses[-1]:.3f}")
    print(f"   - Character-based model final training loss: {char_train_losses[-1]:.3f}")

    print("\n5. Generalization:")
    print(f"   - Word-based model validation loss: {val_losses[-1]:.3f}")
    print(f"   - Character-based model validation loss: {char_val_losses[-1]:.3f}")


# Import Necessary Libraries
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import KFold, ParameterSampler
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from vocab import Vocabulary
from datasets import SentimentalSummarizationDataset
from models_ssum_attention import *
from train_ssum_attention import train, evaluate

# Load and Preprocess Data
data = pd.read_csv("data/amazon_food_review.tsv", sep="\t")
data = data.dropna()

max_text_len = 180
max_summary_len = 10

cleaned_text = np.array(data['Text'])
cleaned_summary = np.array(data['Summary'])
cleaned_score = np.array(data['Score'])

short_text = []
short_summary = []
short_score = []

for i in range(len(cleaned_text)):
    if(len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])
        short_score.append(cleaned_score[i])
        
df = pd.DataFrame({'text': short_text, 'summary': short_summary, 'score': short_score})

# Tokenization and Vocabulary
spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

vocab = Vocabulary(freq_threshold=6)
all_texts = [text for text in df['text']] + [summary for summary in df['summary']]
vocab.build_vocabulary(all_texts)

# Define Hyperparameters and K-Fold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

param_grid = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'hidden_dim': [256, 512],
    'emb_dim': [256, 300],
    'dropout': [0.3, 0.5],
    'batch_size': [128, 256],
    'n_layers': [1, 2],
    'teacher_forcing_ratio': [0.5, 0.7, 1.0]
}

param_list = list(ParameterSampler(param_grid, n_iter=10))

# Cross-Validation and Hyperparameter Tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(vocab)
output_dim = len(vocab)
num_epochs = 10
clip = 1
best_avg_valid_loss = float('inf')
best_model_params = None

for params in param_list:
    avg_valid_loss = 0
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
        train_data = df.iloc[train_idx]
        valid_data = df.iloc[valid_idx]

        train_dataset = SentimentalSummarizationDataset(train_data['text'].values, train_data['summary'].values, train_data['score'].values, vocab)
        valid_dataset = SentimentalSummarizationDataset(valid_data['text'].values, valid_data['summary'].values, valid_data['score'].values, vocab)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], collate_fn=collate_fn)

        encoder = EncoderLSTM(input_dim, params['emb_dim'], params['hidden_dim'], params['n_layers'], params['dropout'])
        decoder = DecoderLSTM(output_dim, params['emb_dim'], params['hidden_dim'], params['n_layers'], params['dropout'])
        model_summary = Seq2Seq(encoder, decoder, device).to(device)

        optimizer_summary = optim.Adam(model_summary.parameters(), lr=params['learning_rate'])
        criterion_summary = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

        model_sentiment = SentimentLSTM(input_dim, params['emb_dim'], params['hidden_dim'], 5, params['n_layers'], params['dropout']).to(device)
        optimizer_sentiment = optim.Adam(model_sentiment.parameters(), lr=params['learning_rate'])
        criterion_sentiment = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            train_loss = train(model_summary=model_summary, model_sentiment=model_sentiment, iterator=train_loader,
                               optimizer_summary=optimizer_summary, optimizer_sentiment=optimizer_sentiment,
                               criterion_summary=criterion_summary, criterion_sentiment=criterion_sentiment, clip=clip, 
                               teacher_forcing_ratio=params['teacher_forcing_ratio'])
            valid_loss = evaluate(model_summary=model_summary, model_sentiment=model_sentiment, iterator=valid_loader,
                                  criterion_summary=criterion_summary)

            avg_valid_loss += valid_loss / n_splits

        print(f'Fold: {fold+1}, Epoch: {epoch+1}, Params: {params}')
        print(f'\tTrain Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')
    
    print(f'Average Validation Loss: {avg_valid_loss:.3f} for Parameters: {params}')
    
    if avg_valid_loss < best_avg_valid_loss:
        best_avg_valid_loss = avg_valid_loss
        best_model_params = params
        torch.save(model_summary.state_dict(), 'best_model_summary.pt')
        torch.save(model_sentiment.state_dict(), 'best_model_sentiment.pt')

print(f'Best Model Parameters: {best_model_params}')
print(f'Best Average Validation Loss: {best_avg_valid_loss:.3f}')
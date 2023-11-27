import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import spacy
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import pandas as pd
from vocab import Vocabulary
from datasets import TextSummarizationDataset
from models_baseline import EncoderLSTM, DecoderLSTM, Seq2Seq

data = pd.read_csv("data/amazon_food_review.tsv", sep="\t")
data = data.dropna()

max_text_len = 180
max_summary_len = 10

cleaned_text =np.array(data['Text'])
cleaned_summary=np.array(data['Summary'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])
        
df=pd.DataFrame({'text':short_text,'summary':short_summary})

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

train_data, temp_data = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0, shuffle=True)

vocab = Vocabulary(freq_threshold=6)
all_texts = [text for text in df['text']] + [summary for summary in df['summary']]
vocab.build_vocabulary(all_texts)

train_dataset = TextSummarizationDataset(train_data['text'].values, train_data['summary'].values, vocab)
valid_dataset = TextSummarizationDataset(valid_data['text'].values, valid_data['summary'].values, vocab)
test_dataset = TextSummarizationDataset(test_data['text'].values, test_data['summary'].values, vocab)

def collate_fn(batch):
    originals, summaries = zip(*batch)
    originals_padded = pad_sequence(originals, padding_value=vocab.stoi["<PAD>"])
    summaries_padded = pad_sequence(summaries, padding_value=vocab.stoi["<PAD>"])

    return originals_padded, summaries_padded

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:-1])
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()  
    epoch_loss = 0

    with torch.no_grad():  
        for i, batch in enumerate(tqdm(iterator)):
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:-1], 0) 

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

device = torch.device("cuda")
input_dim = len(vocab)
output_dim = len(vocab)
encoder = EncoderLSTM(input_dim, 256, 512, 2, 0.5)
decoder = DecoderLSTM(output_dim, 256, 512, 2, 0.5)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters())
pad_idx = vocab.stoi["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

num_epochs = 10
clip = 1

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_loader, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')
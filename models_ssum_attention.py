import torch
import torch.nn as nn 
import torch.nn.functional as F
import random

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)
    
    def hidden(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        return hidden

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, device):
        super(SimpleAttention, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([self.embed_dim])).to(self.device)

    def forward(self, query, key, value):
        # 쿼리와 키의 내적을 계산하여 유사도 점수를 구함
        query, key, value = query.to(self.device), key.to(self.device), value.to(self.device)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # 소프트맥스를 통해 정규화된 어텐션 가중치를 구함
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 가중치와 밸류를 곱하여 출력을 구함
        output = torch.matmul(attention_weights, value)

        return output
    
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
         # Word embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn_text = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=False)
        self.rnn_sent = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=False)

        # Mean-pooling layer
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim) 


    def forward(self, src):
         # Input review is a list of word indices

        # Word embedding
        embedded = self.dropout(self.embedding(src))
        outputs_text, hidden_text = self.rnn_text(embedded)
        outputs_sent, hidden_sent = self.rnn_sent(embedded)

        # Mean-pooling to obtain initial representations
        #v_context = torch.mean(h_context, dim=1)
        #v_sentiment = torch.mean(h_sentiment, dim=1)
        
        #return v_context, self.fc(v_sentiment)
        hidden_sent = hidden_sent[-1, :, :]
        return outputs_text, hidden_text, self.fc(hidden_sent), hidden_sent
    
class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=False)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #encoder_outputs, hidden, sent_outputs = self.encoder(src)

        encoder_outputs, hidden, sent_outputs, hidden_sent = self.encoder(src)
        hidden = 0.7 * hidden + 0.3 * hidden_sent
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs, sent_outputs
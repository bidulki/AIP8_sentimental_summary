import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
    
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
         # Word embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm_context = nn.LSTM(emb_dim, hidden_dim, dropout=dropout)
        self.lstm_sentiment = nn.LSTM(emb_dim, hidden_dim, dropout=dropout)

    def forward(self, src):
        embedded = self.embedding(src)
        
        _, (context_hidden, _) = self.lstm_context(embedded)
        _, (sentiment_hidden, _) = self.lstm_sentiment(embedded)
        
        return context_hidden, sentiment_hidden
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Uc = nn.Linear(hidden_dim, 1)

    def forward(self, context, sentiment):
        combined = torch.cat([context, sentiment], dim=2)
        scores = self.Uc(torch.tanh(self.Wc(combined)))
        attention_weights = F.softmax(scores, dim=0)
        
        return attention_weights

class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context, sentiment):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        
        attention_weights = self.attention(context, sentiment)
        context_vector = torch.sum(attention_weights * context, dim=0, keepdim=True)
        
        # Unsqueeze context_vector to match the sequence length dimension of output
        output = output.unsqueeze(0)

        output = torch.cat([output, context_vector], dim=2)
        output = self.fc(output)
        output = self.dropout(output)
        
        return output, hidden, attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, hidden_dim, output_dim, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src, trg):

        context, sentiment = self.encoder(src)
        target_length = trg.size(0)
        batch_size = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)
        hidden = None
        input = trg[0, :]

        #print(outputs.size())
        for t in range(1, target_length):
            #decoder_input = trg[t]
            
            output, hidden, _ = self.decoder(input, hidden, context, sentiment)
            #print(outputs.size())
            #print(output.size())
            outputs[t] = output

        sentiment = sentiment[-1, :, :]
        return outputs, self.fc(sentiment)
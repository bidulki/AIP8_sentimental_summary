import torch
from tqdm import tqdm

def decode_sequence(sequence):
    decoded_words = []
    for idx in sequence:
        word = y_voc.itos[idx.item()]

        if word == "<SOS>":
            continue

        if word == "<EOS>":
            break

        if word != "<PAD>":
            decoded_words.append(word)

    decoded_sentence = ' '.join(decoded_words)
    return decoded_sentence

def decode(model_summary, model_sentiment, iterator, device, y_voc):
    decoded_sentences = []
    model_summary.eval()
    model_sentiment.eval()
    trg_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src, trg, label = batch
            for i in range(trg.shape[1]):
                trg_list.append(decode_sequence(trg[:,i],y_voc))
            src, trg, label = src.to(device), trg.to(device), label.to(device)
            label = label.squeeze()
            sentiment_hidden = model_sentiment.hidden(src)
            output = model_summary(src, trg, sentiment_hidden, 0)

            for i in range(output.shape[1]):
                predict = output.argmax(dim=2)[:,i]
                decoded = decode_sequence(predict,y_voc)
                decoded_sentences.append(decoded)

    return decoded_sentences, trg_list


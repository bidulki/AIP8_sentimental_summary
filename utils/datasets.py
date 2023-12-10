import torch
from torch.utils.data import Dataset

class SentimentalSummarizationDataset(Dataset):
    def __init__(self, original_texts, summaries, labels, x_voc, y_voc):
        self.original_texts = original_texts
        self.summaries = summaries
        self.labels = labels
        self.x_voc = x_voc
        self.y_voc = y_voc

    def __len__(self):
        return len(self.original_texts)

    def __getitem__(self, index):
        original_text = self.original_texts[index]
        summary = self.summaries[index]
        label = self.labels[index]

        numericalized_original = [self.x_voc.stoi["<SOS>"]]
        numericalized_original += self.x_voc.numericalize(original_text)
        numericalized_original.append(self.x_voc.stoi["<EOS>"])

        numericalized_summary = [self.y_voc.stoi["<SOS>"]]
        numericalized_summary += self.y_voc.numericalize(summary)
        numericalized_summary.append(self.y_voc.stoi["<EOS>"])

        return torch.tensor(numericalized_original), torch.tensor(numericalized_summary), torch.tensor([label-1], dtype=torch.long)
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd

class CoherenceScoreDataset(Dataset):

    def __init__(self, filename):
        
        self.pos_context, self.neg_context, self.labels = [], [], []

        data = defaultdict(list)
        
        with open(filename) as f:
            for fline in f:
                id, para = fline.strip().split('\t')
                sents = para.strip().split(' @eos@ ')

                data[id].append(sents)

        for item in data.values():
            for i in range(len(item)-1):
                self.pos_context.append(item[i])
                self.neg_context.append(item[i+1])
                self.labels.append(1)


    def __len__(self):
        return len(self.pos_context)

    def __getitem__(self, index): 
        pos_context = self.pos_context[index]
        neg_context = self.neg_context[index]
        label = self.labels[index]
        
        return pos_context, neg_context, label
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def load_dataset(filename, batch_size, shuffle):
    dataset = CoherenceScoreDataset(filename)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader

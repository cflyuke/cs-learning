from torch.utils.data import Dataset, DataLoader
import torch
import os
import re
import pickle

with open('Emo_classify/model/word_sequence.pkl', 'rb') as file:
    WS = pickle.load(file)

FILTERS = ['\t', '\n', '#', '$', '%', '&', '<','>','(',')', '\x97', '\x96', ',', '.',':']
TRAIN_BATCHSIZE = 128
TEST_BATCHSIZE = 128
TRAIN_PATH = r'Emo_classify/Imdb/train'
TEST_PATH = r'Emo_classify/Imdb/test'
MAX_LEN = 100
class IMDBDataset(Dataset):
    def __init__(self, train=True, transform=None,train_path = TRAIN_PATH, test_path = TEST_PATH):
        self.data_path = train_path if train else test_path
        self.total_data_path = []
        self.transform = transform
        temp_data_path =[os.path.join(self.data_path, 'pos'), os.path.join(self.data_path, 'neg')]
        for i in temp_data_path :
            name_list = os.listdir(i)
            self.total_data_path.extend([os.path.join(i, j) for j in name_list])
        
    def __getitem__(self, index):
        file_path = self.total_data_path[index]
        label = 0 if file_path.split('/')[-2]=='neg' else 1
        with open(file_path, 'r') as file:
            content = file.read()
        tokens = tokenization(content)
        if self.transform:
            tokens = self.transform(tokens)
        return tokens, label
    
    def __len__(self):
        return len(self.total_data_path)
    


def tokenization(content):
    escaped_filters = [re.escape(i) for i in FILTERS]
    content = re.sub('|'.join(escaped_filters), ' ', content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


def get_data_loader(train = True):
    imdb_dataset = IMDBDataset(train=train)
    batchsize = TRAIN_BATCHSIZE if train else TEST_BATCHSIZE
    imdb_dataloader = DataLoader(imdb_dataset, batch_size=batchsize, shuffle = True, collate_fn=collate_fn)
    return imdb_dataloader

def collate_fn(batch):
    content, label = list(zip(*batch))
    content = [WS.transform(i, max_len=MAX_LEN) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content, label

if __name__=='__main__':
    Imdb_dataloader = get_data_loader()
    for index, (data, target) in enumerate(Imdb_dataloader):
        print(data)
        print(target)
        break
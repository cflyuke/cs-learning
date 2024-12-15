import torch.nn as nn
import torch
import numpy as np
import Dataset
import os

SEN_LEN = Dataset.MAX_LEN



class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding_dim = 100
        self.hidden_size = 128
        self.num_layers = 2
        self.bidirectional = True
        self.num_di = 2 if self.bidirectional else 1
        self.dropout = 0.5



        self.embedding = nn.Embedding(len(Dataset.WS.word2index), self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_di, 
            batch_first=True, 
            bidirectional=self.bidirectional, 
            dropout=self.dropout
            )
        self.fc1 = nn.Linear(self.hidden_size * self.num_di , 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(True)



    def forward(self, input):
        '''
        Para:
        input:[Batchsize, max_len, EMBED_DIM]
        '''
        out = self.embedding(input)
        out, (h_n, c_n) = self.lstm(out)
        out_f, out_b = h_n[-1,:,:], h_n[-2, :, :]
        out = torch.cat([out_f, out_b], dim = -1)

        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.fc3(out)

        out = nn.functional.log_softmax(out, dim=-1)
        return out
    
def train(epoch):
    for index, (input, target) in enumerate(Dataset.get_data_loader(train=True)):
        output = imbedmodel(input)
        loss = nn.functional.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index%10 == 0:
            print(f'epoch:{epoch} index:{index} loss:{loss.item()}')
    torch.save(imbedmodel.state_dict(),'Emo_classify/model/ImdbModel_m.pkl' )
    torch.save(optimizer.state_dict(),'Emo_classify/model/optimizer_m.pkl')

def test():
    imbedmodel.eval()
    loss_list = []
    accuracy_list = []
    with torch.no_grad():
        for index ,(input, target) in enumerate(Dataset.get_data_loader(train = False)):
            output = imbedmodel(input)
            loss = nn.functional.nll_loss(output, target)
            pred = output.argmax(dim = 1)
            accuracy = pred.eq(target).float().mean()
            loss_list.append(loss)
            accuracy_list.append(accuracy)
        print(f'平均准确率:{np.mean(accuracy_list)*100:.2f}%, 平均误差:{np.mean(loss_list):.4f}' )



if __name__ == '__main__':
    Epoch = 3
    imbedmodel = ImdbModel()
    optimizer = torch.optim.Adam(imbedmodel.parameters(), 0.001)
    if os.path.exists('Emo_classify/model/ImdbModel_m.pkl'):
        imbedmodel.load_state_dict(torch.load('Emo_classify/model/ImdbModel_m.pkl'))
        optimizer.load_state_dict(torch.load('Emo_classify/model/optimizer_m.pkl'))
    for i in range(Epoch):
        train(i)
        test()
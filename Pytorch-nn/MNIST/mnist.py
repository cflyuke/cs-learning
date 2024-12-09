from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
import torch
import os
import numpy as np

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000


#1. 准备数据集
def get_dataloader(batch_size, train=True):
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        # 对于单通道图像，mean 和 std 是单个值的数组；对于多通道图像，mean 和 std 需要包含每个通道的均值和标准差
        transforms.Normalize(mean = [0.1307], std = [0.3081])
    ])
    dataset = MNIST(root = './data', train=train, transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


#2. 构建模型
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)
    
    def forward(self, input):
        x = input.view(input.size(0), -1)
        out1 = self.fc1(x)
        out2 = nn.functional.relu(out1)
        out3 = self.fc2(out2)       
        predict = nn.functional.log_softmax(out3, dim=1)
        return predict


#3. 模型的训练
def train(epoch, train=True):
    mnist.train(mode=train)
    train_dataloader = get_dataloader(TRAIN_BATCH_SIZE, train=train)
    for idx , (data, target) in enumerate(train_dataloader):
        predict = mnist(data)
        loss = nn.functional.nll_loss(predict, target)
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        if idx % 1000 == 0:    
            print(f'Epoch: {epoch}, Batch: {idx}, Loss: {loss.item()}')
            torch.save(mnist.state_dict(), 'model/model_para.pkl')
            torch.save(optimizer.state_dict(), 'model/optim_para.pkl')


#4 模型的评估
def test():
    mnist.eval()
    loss_list = []
    accurate_list = []
    test_dataloader = get_dataloader(TEST_BATCH_SIZE, train=False)
    with torch.no_grad():
        for idx, (input, target) in enumerate(test_dataloader):
                output = mnist(input)
                loss = nn.functional.nll_loss(output, target)
                pred = output.argmax(dim = 1)
                accurate = pred.eq(target).float().mean()
                loss_list.append(loss)
                accurate_list.append(accurate)
    print(f'平均准确率:{np.mean(accurate_list) * 100:.2f}%, 平均损失:{np.mean(loss_list):.4f}')
            




if __name__ == '__main__':
    mnist = MnistNet()
    optimizer = torch.optim.Adam(mnist.parameters(), lr=0.001)
    if os.path.exists('model/optim_para.pkl'):
        optimizer.load_state_dict(torch.load('model/optim_para.pkl'))
        mnist.load_state_dict(torch.load('model/model_para.pkl'))
    num_epochs = 3
    for epoch in range(num_epochs):
        train(epoch)
        test()




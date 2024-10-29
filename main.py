#导入模块
import torch 
import os 
from torchvision import transforms
import random
import gzip
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt 
import torch.nn.functional as F
from PIL import Image
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#使用torchvisiom.transform将图片转换为张量

transform =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
#构建数据集（使用MINIST）
path ='./MNIST/raw'

EPOCH =10
Batch_Size = 64
#创建dataset

class Dataset:
    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.train = train
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        if self.train:
            file_name = 'train-images-idx3-ubyte.gz'
            label_file_name = 'train-labels-idx1-ubyte.gz'
        else:
            file_name = 't10k-images-idx3-ubyte.gz'
            label_file_name = 't10k-labels-idx1-ubyte.gz'

        with gzip.open(os.path.join(self.data_path, file_name), 'rb') as f:
            self.images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

        with gzip.open(os.path.join(self.data_path, label_file_name), 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

#创建dataloader
class DataLoader:

    def __init__(self,dataset,batch_size,is_shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

        self.data_indices = list(range(len(dataset)))
    #打乱数据集
    def shuffle_data(self):
        if self.is_shuffle:
            random.shuffle(self.data_indices)
    #迭代数据集
    def __iter__(self):
        self.shuffle_data()
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.data_indices[i:i + self.batch_size]
            batch_images = []
            batch_labels = []
            for idx in batch_indices:
                image, label = self.dataset[idx]
                batch_images.append(image)
                batch_labels.append(label)
            yield (batch_images, batch_labels)
#下载数据集
#下载训练集
trainData=Dataset(path,train=True)
#下载测试集
testData=Dataset(path,train=False)
#shuffle=true是用来打乱数据集的
train_Dataloader = DataLoader(trainData,batch_size=Batch_Size,is_shuffle=True)
test_DataLoader = DataLoader(testData,batch_size=Batch_Size,is_shuffle=False)
class Net(torch.nn.Module):
    #构造函数
    def __init__(self):
        #继承父类
        super(Net,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )
    def forward(self,x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x

EPOCH = 100
model = Net().to(device)
#使用交叉墒损失做损失函数
sunshi = torch.nn.CrossEntropyLoss()
#优化器：随机梯度下降
#lr=学习率，momentum = 冲量
optimizer = torch.optim.SGD(model.parameters(),lr=0.25,momentum=0.25)
#训练
def train(epoch):
    running_loss = 0.0
    running_total=0
    running_correct = 0
    for batch_idx, (inputs, target) in enumerate(train_Dataloader, 0):
        inputs = torch.stack([torch.tensor(img, dtype=torch.float) for img in inputs]).to(device)
        target = torch.tensor(target, dtype=torch.long).to(device)
        #梯度归零
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = sunshi(outputs,target)
        #反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #准确率
        _,predicted = torch.max(outputs,dim=1)
        running_total+=inputs.shape[0]
        running_correct += (predicted == target).sum().item()
        print('[%d,%5d]:loss:%.3f,acc:%.2f',epoch+1,batch_idx+1,running_loss,running_correct/running_total)
#测试
def test():
    correct =0
    total = 0
    with torch.no_grad():
        for data in test_DataLoader:
            images,labels = data
            outputs = model(images)
            predicted = torch.max(outputs.data,dim=1)
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()
    accuracy = correct/total
    print('[%d/%d]Accuracy: %.lf %%', epoch+1,EPOCH,accuracy)
    return accuracy



if __name__ =='__main__':
    acc_list_test =[]
    for epoch in range(EPOCH):
        train(epoch)
        #每训练10轮测试一次
        if epoch % 10 ==9:
            acc_test = test()
            acc_list_test.append(acc_test)
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
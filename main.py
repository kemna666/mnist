#导入模块
import torch 
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import matplotlib

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#使用torchvisiom.transform将图片转换为张量
transform =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5],std=[0.5])])
#构建数据集（使用MINIST）
path ='./MNIST'
#下载数据集
#下载训练集
trainData=torchvision.datasets.MNIST(path,train=True,transform=transform,download=True)
#下载测试集
testData=torchvision.datasets.MNIST(path,train=False,transform=transform,download=False)
Batch_Size = 300
#shuffle=true是用来打乱数据集的
train_Dataloader = DataLoader(trainData,batch_size = Batch_Size,shuffle = True)
test_DataLoader = DataLoader(testData,batch_size=Batch_Size,shuffle=False)
class Net(torch.nn.Module):
    #构造函数
    def __init__(self):
        #继承父类
        super(Net,self).__init__()
        self.conv1=torch.nn.Sequential(
            #torch.nn.Conv2d(in_channels,out_channels,卷积核大小,步长,填充)
            torch.nn.Conv2d(1,10,kernel_size=5),
            #torch.nn.RelU()激活函数
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=x.view(Batch_Size,-1)
        x=self.fc(x)
        return x

model = Net()
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
    for batch_idx,data in enumerate(train_Dataloader,0):
        inputs,target = data
        #梯度归零
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs,target)
        #反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #准确率
        predicted = torch.max(outputs.data,dim=1)
        running_total+=inputs.shape[0]
        running_correct +=(predicted == target).sum().item()
        print('[%d,%5d]:loss:%.3f,acc:%.2f',epoch+1,batch_idx+1,running_loss,running_correct/running_total)
#测试
def test():
    correct =0
    total = 0
    #
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

#
EPOCH =10

if __name__ ==' __main__':
    acc_list_test =[]
    for epoch in range(EPOCH):
        train(epoch)
        #每训练10轮测试一次
        if epoch % 10 ==9:
            acc_test = test()
            acc_list_test.append(acc_test)
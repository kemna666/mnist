#导入模块
import torch 
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from tqdm import tqdm
import matplotlib

#进行数据预处理
#初始化存储训练和测试损失与准确率的字典
history = {
    'Train Loss': [],
    'Train Accuracy': [],
    'Test Loss': [],
    'Test Accuracy': []
}
#将训练设备设置为GPU
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
#使用dataloader方法开始训练
#设定batch大小
BATCH_SIZE=2048
#构建dataloader
TrainDataLoader = torch.utils.data.DataLoader(dataset = trainData,batch_size=BATCH_SIZE)
TestDataLoader = torch.utils.data.DataLoader(dataset=testData,batch_size=BATCH_SIZE)
#构建神经网络
class Net(torch.nn.Module):
    #构造函数
    def __init__(self):
        #继承父类
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
        torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 14x14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            #The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7 * 7 * 64,out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128,out_features = 10),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self,input):
        output = self.model(input)
        return output
net = Net().to(device)
#构建迭代器与损失函数
#对于简单的多分类任务，我们可以使用交叉熵损失来作为损失函数；而对于迭代器而言，我们可以使用Adam迭代器
lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
#循环训练
Epochs=100
for epochs in range(0,Epochs):
    #训练内容
    processBar = tqdm(TrainDataLoader,unit = 'step')
    net.train(True)
    for step,(trainImgs,labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)
        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs,labels)
        predictions = torch.argmax(outputs, dim = 1)
        accuracy = torch.sum(predictions == labels)/labels.shape[0]
        loss.backward()
        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epochs,Epochs,loss.item(),accuracy.item()))
        
        if step == len(processBar)-1:
            correct,totalLoss = 0,0
            net.train(False)
            for testImgs,labels in TestDataLoader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                loss = lossF(outputs,labels)
                predictions = torch.argmax(outputs,dim = 1)
                
                totalLoss += loss
                correct += torch.sum(predictions == labels)
            testAccuracy = correct/(BATCH_SIZE * len(TestDataLoader))
            testLoss = totalLoss/len(TestDataLoader)
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % 
                                   (epochs,Epochs,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    processBar.close()
torch.save(net,'./model.pth')

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from visdom import Visdom
from utils import InformationEntropy
import numpy as np
import time
print(time.asctime(time.localtime(time.time())))
seed = 26535
torch.manual_seed(seed)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
batchsz = 128
lr = 1e-2

epochs = 80
viz = Visdom()

transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10('../data', transform=transform)
eval_dataset = datasets.CIFAR10('../data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batchsz, shuffle=True)
# vgg16
teacher = models.vgg16(pretrained=True)
teacher.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )
teacher.load_state_dict(torch.load('model/vgg16_pretrained/vgg16_40'))
teacher.eval()
# vgg11
net = models.vgg11()
net.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
net.to(device)
teacher.to(device)
criterion.to(device)
total_loss = []

viz.line([[0.,0.]], [0.], win='train&&eval loss', opts=dict(title='train&&eval loss',
                                                            legend=['train', 'eval']))
viz.line([0.], [0.], win='accuracy', opts=dict(title='accuracy'))
for epoch in range(epochs):
    net.train()
    total_loss.clear()
    for step, (input, _) in enumerate(train_loader):
        input = input.to(device)
        target = teacher(input)
        target = torch.softmax(target, dim=-1).detach()
        logits = net(input)
        prob = torch.softmax(logits, dim=-1)
        loss = InformationEntropy(prob, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        if step%50==0:
            print('epoch:{} batch:{} loss:{}'.format(epoch, step, loss.item()))
    scheduler.step()
    eval_loss, correct = 0, 0
    net.eval()
    for input, target in eval_loader:
        input, target = input.to(device), target.to(device)
        logits = net(input)
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
        loss = criterion(logits, target)
        eval_loss += input.shape[0] * loss.item()
    acc = 100 * correct / len(eval_loader.dataset)
    eval_loss /= len(eval_loader.dataset)
    viz.line([[float(np.mean(total_loss)), eval_loss]], [epoch], win='train&&eval loss', update='append')
    viz.line([acc], [epoch], win='accuracy', update='append')
    print('epoch:{} loss:{} acc:{:.2f}%'.format(epoch, eval_loss, acc))
    # torch.save(net.state_dict(), 'model/vgg16_pretrained/vgg16_{}'.format(epoch))






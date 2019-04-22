# import packages
import sys
import cv2
import torch
import numpy as np
import matplotlib
#%matplotlib inline
import PIL.Image as image
import matplotlib.pyplot as plt

import torch
import torchvision

print(torch.__version__)

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
torch.cuda.set_device(1)
from time import *
import pdb


# training hyperparameters
total_epoch = 15
pre_epoch = 0
batch_size = 192
learning_rate = 0.01
gpu_id = 1


#### data
# train_loader = torch.utils.data.DataLoader([train_data, train_labels], batch_size=batch_size, shuffle=True, num_workers=2)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="/home/ltq/cifar/", train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)



##### model
# -resnet18
resnet18 = models.resnet18(pretrained=False)
resnet18.fc = nn.Linear(512, 10)

# vgg16 = models.vgg16(pretrained=False)
# vgg16.fc = nn.Linear(1000, 10)

model = resnet18.cuda(gpu_id)



# loss and optimizer
getloss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# training
best_acc = 50
print("Starting training")
start_time = time()
for epoch in range(pre_epoch, total_epoch):
    print("epoch: {}".format(epoch+1))
    start_epoch = time()
    model.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(train_loader, 0):
        length = len(train_loader)
        inputs, labels = data
        inputs = torch.autograd.Variable(inputs).cuda(gpu_id)
        labels = torch.autograd.Variable(labels).cuda(gpu_id)
#         print(inputs.shape, labels.shape)

        optimizer.zero_grad()
        
        # forward + backward
        outputs = model(inputs)
        loss = getloss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print loss and accuracy every epoch
#         pdb.set_trace()
        sum_loss += float(loss.data.cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%03d, iter:%05d] Loss: %.03f | Acc: %.3f%% | time: %.02fs '
          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, time() - start_epoch))
    torch.save(model.state_dict(), open("state_{}_batch{}_epoch{}.pth".format("resnet18", batch_size, epoch+1), "w"))
    print("epoch %03d, used time total: %.2f" % (epoch+1, time() - start_time))

# classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy

from models import Xception

# Xception
model = Xception().cuda()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                shuffle=True, num_workers=8)

# test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                shuffle=False, num_workers=8)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

    torch.save(deepcopy(model).cpu().state_dict(), 'model_data/model'+str(epoch)+'.pth')


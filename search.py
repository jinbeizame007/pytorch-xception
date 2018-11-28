import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models import Xception

model = Xception()
model.load_state_dict(torch.load('model_data/model9.pth'))
model.cuda()
model.eval()

cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
cos_sim = cos_sim.cuda()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                shuffle=False, num_workers=1)

# test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                shuffle=False, num_workers=8)


train_data, train_label = [data for data in trainloader][0]
train_data, train_label = train_data.cuda(), train_label.numpy()

test_data, test_label = [data for data in testloader][0]
test_data, test_label = test_data.cuda(), test_label.numpy()

train_embed = model.embedding(train_data)
test_embed = model.embedding(test_data)

accuracy = 0.
for i in range(100):
    similarity = cos_sim(test_embed[i].unsqueeze(0), train_embed).detach().cpu().numpy()
    sorted_sim_indexes = np.argsort(similarity)[::-1]
    count = 0
    for j in range(10):
        if train_label[sorted_sim_indexes[j]] == test_label[i]:
            count += 1
    accuracy += count / 10.
accuracy /= 100.
print(accuracy)
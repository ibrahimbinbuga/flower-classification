import torch
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from utils.train import train_model

def vgg16_feature_extractor(train_loader, num_epochs=3, device="cpu"):
    vgg16 = models.vgg16(pretrained=True)

    for param in vgg16.parameters():
        param.requires_grad = False

    vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 5)

    vgg16 = vgg16.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001, momentum=0.9)

    vgg16 = train_model(vgg16, train_loader, criterion, optimizer, num_epochs, device)

    return vgg16

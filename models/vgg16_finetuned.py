import torch
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from utils.train import train_model

def vgg16_finetuned(train_loader, num_epochs=3, device="cpu"):
    vgg16_finetuned = models.vgg16(pretrained=True)

    for param in vgg16_finetuned.features[:6].parameters():
        param.requires_grad = False

    vgg16_finetuned.classifier[6] = nn.Linear(vgg16_finetuned.classifier[6].in_features, 5)

    vgg16_finetuned = vgg16_finetuned.to(device)

    optimizer_finetune = optim.SGD(vgg16_finetuned.parameters(), lr=0.0001, momentum=0.9)
    criterion_finetune = nn.CrossEntropyLoss()

    vgg16_finetuned = train_model(vgg16_finetuned, train_loader, criterion_finetune, optimizer_finetune, num_epochs, device)

    return vgg16_finetuned

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        for param in self.inception.parameters():
           param.requires_grad = False

        self.inception.fc = nn.Linear(2048, nclasses)

        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(2048, nclasses)

    def forward(self, x): 
        return self.resnet(x) + self.inception(x)
import torch
from pascal_voc_dataset import PascalVOCDataset
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ResNet50(nn.Module):
    def __init__(self, hidden_size:int, output_size:int, transfer:bool):
        super(ResNet50, self).__init__()

        self.pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, hidden_size)

        if transfer:
            self.freeze_pretrained_weights()

        self.model = nn.Sequential(
            self.pretrained_model,
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_pretrained_weights(self):
        for name, param in self.pretrained_model.named_parameters():
            if 'fc' in name:
                continue
            param.requires_grad = False

    def unfreeze_pretrained_weights(self):
        for name, param in self.pretrained_model.named_parameters():
            if 'fc' in name:
                continue
            param.requires_grad = True

if __name__ == '__main__':
    model = ResNet50(hidden_size=100, output_size=20)
    print(model)
    conv1_first_block = model.pretrained_model.conv1
    print(conv1_first_block.weight.size(0))

import torch
from pascal_voc_dataset import PascalVOCDataset
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torch import nn, optim

from model import *
from trainer import *
from loss import *

# DATA
train_dataset = PascalVOCDataset(images_path='pascal_voc_2007_data/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/',
                                 labels_path='labels/processed_train_labels.csv',
                                 preprocess=ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=False))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = PascalVOCDataset(images_path='pascal_voc_2007_data/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/',
                               labels_path='labels/processed_val_labels.csv',
                               preprocess=ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=False))
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# MODEL
model = ResNet50(hidden_size=128, output_size=20, transfer=False)

# TRAINER
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = WeightedBCELoss(scale=10, reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

trainer = Trainer(model, train_dataloader, val_dataloader,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  device=device)

trainer.train(num_epochs=10)

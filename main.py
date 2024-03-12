import torch
from pascal_voc_dataset import PascalVOCDataset
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torch import nn, optim

from model import *
from trainer import *
from loss import *

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Input batch size', default=32)
    parser.add_argument('--hidden_size', help='Input hidden size', default=128)
    parser.add_argument('--transfer', help='Choose whether to freeze pretrained weights', default=False)
    parser.add_argument('--pos_scale', help='Input positive label scale', default=5)
    parser.add_argument('--learning_rate', help='Input learning rate', default=0.0001)
    parser.add_argument('--step_size', help='Input step size', default=10)
    parser.add_argument('--gamma', help='Input gamma', default=0.3)
    parser.add_argument('--num_epochs', help='Input number of epochs', default=10)
    args = parser.parse_args()

    # DATA
    train_dataset = PascalVOCDataset(images_path='pascal_voc_2007_data/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/',
                                    labels_path='labels/processed_train_labels.csv',
                                    preprocess=ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=False))
    train_dataloader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)

    val_dataset = PascalVOCDataset(images_path='pascal_voc_2007_data/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/',
                                labels_path='labels/processed_val_labels.csv',
                                preprocess=ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=False))
    val_dataloader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False)

    # MODEL
    model = ResNet50(hidden_size=int(args.hidden_size), output_size=20, transfer=args.transfer)

    # TRAINER
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = WeightedBCELoss(scale=int(args.pos_scale), reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate), betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.step_size), gamma=float(args.gamma))

    trainer = Trainer(model, train_dataloader, val_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device)

    trainer.train(num_epochs=int(args.num_epochs))

if __name__ == '__main__':
    main()

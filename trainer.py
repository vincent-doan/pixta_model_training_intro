import time, gc
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, loss_fn=None, optimizer=None, scheduler=None, device=None, continue_training=False):
        # MAIN COMPONENTS
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        
        # METRICS (threshold=0.5, average='macro')
        num_labels = train_dataloader.dataset.num_labels
        self.train_accuracy = MultilabelAccuracy(num_labels=num_labels).to(device)
        self.val_accuracy = MultilabelAccuracy(num_labels=num_labels).to(device)
        
        self.train_precision = MultilabelPrecision(num_labels=num_labels,average=None).to(device)
        self.val_precision = MultilabelPrecision(num_labels=num_labels, average=None).to(device)

        self.train_recall= MultilabelRecall(num_labels=num_labels, average=None).to(device)
        self.val_recall = MultilabelRecall(num_labels=num_labels, average=None).to(device)
        
        self.train_f1_score = MultilabelF1Score(num_labels=num_labels, average=None).to(device)
        self.val_f1_score = MultilabelF1Score(num_labels=num_labels, average=None).to(device)

        # OTHER SETUP
        self.continue_training = continue_training
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.last_valid_loss = float('inf')
        self.total_train_samples = len(train_dataloader.dataset)
        self.total_val_samples = len(val_dataloader.dataset)
        self.writer = SummaryWriter()

    def save_model(self, epoch, train_loss, valid_loss, checkpoint_path):
        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['valid_loss']

    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        preds = self.model(inputs)
        
        loss = self.loss_fn(preds, targets)
        self.train_accuracy.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_recall.update(preds, targets)
        self.train_f1_score.update(preds, targets)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        preds = self.model(inputs)
        
        loss = self.loss_fn(preds, targets)
        self.val_accuracy.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_f1_score.update(preds, targets)

        return loss.item()

    def adjust_learning_rate(self, new_scheduler):
        self.scheduler = new_scheduler

    def train(self, num_epochs:int):
        self.model.to(self.device)

        for epoch in range(num_epochs):
            start = time.time()

            # -------------------- TRAIN LOOP -------------------- #
            self.model.train()
            train_epoch_loss = 0
            for inputs, targets in tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                loss = self.train_step(inputs, targets)
                train_epoch_loss += loss

            del inputs
            del targets
            gc.collect()
            torch.cuda.empty_cache()

            train_epoch_loss /= self.total_train_samples
            self.writer.add_scalar('Loss/train', train_epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/train', self.train_accuracy.compute().item(), epoch)
            self.writer.add_text('Precision/train', str(self.train_precision.compute()), epoch)
            self.writer.add_text('Recall/train', str(self.train_recall.compute()), epoch)
            self.writer.add_text('F1 score/train', str(self.train_f1_score.compute()), epoch)

            end = time.time()

            # -------------------- VALID LOOP -------------------- #
            self.model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                for inputs, targets in tqdm(self.val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                    loss = self.val_step(inputs, targets)
                    val_epoch_loss += loss
            
            del inputs
            del targets
            gc.collect()
            torch.cuda.empty_cache()

            val_epoch_loss /= self.total_val_samples
            self.writer.add_scalar('Loss/val', val_epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/val', self.val_accuracy.compute().item(), epoch)
            self.writer.add_text('Precision/val', str(self.val_precision.compute()), epoch)
            self.writer.add_text('Recall/val', str(self.val_recall.compute()), epoch)
            self.writer.add_text('F1 score/val', str(self.val_f1_score.compute()), epoch)

            # -------------------- VISUALIZE FILTERS -------------------- #
            first_conv_layer = self.model.pretrained_model.conv1
            num_filters = first_conv_layer.weight.size(0)
            first_conv_layer_weights = first_conv_layer.weight.data
            
            weights_min, weights_max = first_conv_layer_weights.min(), first_conv_layer_weights.max()
            normalized_weights = (first_conv_layer_weights - weights_min) / (weights_max - weights_min)
            fig, axs = plt.subplots(num_filters // 8, 8, figsize=(12, 12))

            for i in range(num_filters // 8):
                for j in range(8):
                    axs[i, j].imshow(normalized_weights[i * 8 + j, 0].cpu(), cmap='viridis')
                    axs[i, j].axis('off')

            filter_visualization_path = 'filters/filters_visualization_{0}.png'.format(epoch + 1)
            plt.savefig(filter_visualization_path)
            self.writer.add_image('Filters/FirstConvLayer', plt.imread(filter_visualization_path), epoch, dataformats='HWC')

            # -------------------- METRICS -------------------- # 
            self.writer.add_scalar('Time/train', round(end-start, 2), epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'])
            self.scheduler.step()

            # -------------------- RESET METRICS -------------------- #
            self.train_accuracy.reset()
            self.val_accuracy.reset()
            self.train_precision.reset()
            self.val_precision.reset()
            self.train_recall.reset()
            self.val_recall.reset()
            self.train_f1_score.reset()
            self.val_f1_score.reset()

            # -------------------- CHECKPOINT -------------------- #
            self.save_model(epoch, train_epoch_loss, val_epoch_loss, 'checkpoint/last_model_checkpoint.pth')
            if val_epoch_loss < self.last_valid_loss:
                self.last_valid_loss = val_epoch_loss
                self.save_model(epoch, train_epoch_loss, val_epoch_loss, "checkpoint/best_model_checkpoint.pth")
                self.writer.add_text('Checkpoint', 'New checkpoint saved.', epoch)
        
        self.writer.flush()
        self.writer.close()
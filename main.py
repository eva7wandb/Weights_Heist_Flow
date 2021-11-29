from utils import (
    setup, data, viz
)
from torch_lr_finder import LRFinder
from utils.training import train
from utils.testing import test

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

setup.set_seed()
cuda = setup.is_cuda()
device = setup.get_device()


class Trainer:
    def __init__(
        self, model,
        lr=0.01,
        batch_size=128,
        scheduler = 'ReduceLROnPlateau',  #values are CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
        model_viz=True,
        model_path=None,
        eval_model_on_load=True,
        label_smoothing=0.0,
        optimizer='SGD',
        run_find_lr=False
    ):
        print(f"[INFO] Loading Data")
        self.train_loader = data.CIFAR10_dataset(
            train=True, cuda=cuda
        ).get_loader(batch_size)
        self.test_loader = data.CIFAR10_dataset(
            train=False, cuda=cuda
        ).get_loader(batch_size)
        self.test_loader_unnormalized = data.CIFAR10_dataset(
            train=False, cuda=cuda, normalize=False
        ).get_loader(batch_size)

        self.net = model.to(device)
        if model_viz:
            viz.show_model_summary(self.net)
        
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        if optimizer=='SGD':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=1e-4
            )
        elif optimizer=='Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=self.lr, 
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
            )
        else:
            raise ValueError(f'{optimizer} is not valid choice. Please select one of valid scheduler - SGD, Adam')
        
        print(self.optimizer)
        print('-' * 64)
        if run_find_lr:
            print('Running LR finder ... ')
            self.find_lr()
        print('-' * 64)
        
        if scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        elif scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        elif scheduler == 'OneCycleLR':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr, steps_per_epoch=len(self.train_loader), 
                epochs=24, div_factor=10, 
                final_div_factor=1, pct_start=0.2, 
                three_phase=False, anneal_strategy='linear'
            )
        else:
            raise ValueError(f'{scheduler} is not valid choice. Please select one of valid scheduler - CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau')

        self.logs = []
        self.lr_logs = []
        
        self.model_path = model_path
        if self.model_path:
            self.load_model()
        
        if eval_model_on_load:
            self.evaluate_model()
    
    def load_model(self):
        print("[INFO] Loading model to from path {}".format(self.model_path))
        self.net.load_state_dict(torch.load(self.model_path))
        
    def train_model(self, epochs, wandb=None):
        EPOCHS = epochs
        print(f"[INFO] Begin training for {EPOCHS} epochs.")
        
        for epoch in range(EPOCHS):
            lr_at_start_of_epoch = self.optimizer.param_groups[0]['lr']
            
            train_batch_loss, train_batch_acc, train_batch_lrs = train(
                self.net, device, 
                self.train_loader, self.optimizer, self.criterion, epoch, self.scheduler,
            )
            train_loss = np.mean(train_batch_loss)
            train_acc = np.mean(train_batch_acc)
            test_loss, test_acc = test(
                self.net, device,
                self.test_loader, self.criterion, epoch,
            )
            self.lr_logs.extend(train_batch_lrs)
            #self.scheduler.step(test_loss)
            
            ## logging
            log_temp = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "lr": lr_at_start_of_epoch,
            }
            try:
                wandb.log(log_temp)
            except:
                pass
            
            self.logs.append(log_temp)
    
    def evaluate_model(self):
        test_loss, test_acc = test(
            self.net, device,
            self.test_loader, self.criterion, epoch=0,
        )
    
    def find_lr(self, init_value=1e-8, final_value=8.0):
        print(f'Starting Find_LR Algorithm')
        print(f'initial_LR from model, self.lr is {self.lr}')
        number_in_epoch = len(self.train_loader) - 1
        print(f'number_in_epoch is {number_in_epoch}')
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        print(f'update_step is {update_step})
        self.lr = init_value
        print(f'after setting int_value, model self.lr is {self.lr}')
        self.optimizer.param_groups[0]["lr"] = self.lr
        print(f'Confirming that the self.lr passes onto optimizer.param_groups: {self.optimizer.param_groups[0]["lr"]}')
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        for data in self.train_loader:
            batch_num += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            # exit if loss shoots up
            if batch_num > 1 and loss > 4 * best_loss:
                return log_lrs[10:-5], losses[10:-5]

            # Record the best loss
            if loss < best_loss or batch_num == 1:
                best_loss = loss

            # Store the values
            losses.append(loss)
            log_lrs.append(math.log10(self.lr))

            # Do the backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # update the lr for next step and store
            self.lr *= update_step
            print(f'after update, new model self.lr becomes {self.lr}')
            
            self.optimizer.param_groups[0]["lr"] = self.lr
            print(f'Confirming that the self.lr passes onto optimizer.param_groups: {self.optimizer.param_groups[0]["lr"]}']
        return log_lrs[10:-5], losses[10:-5]


def show_misclassification(trainer, cam_layer_name='layer4'):
    from utils.viz import visualize_sample
    from utils.testing import get_sample_predictions
    
    sample_preds = get_sample_predictions(trainer)
    
    for class_, samples in sample_preds['mistakes'].items():
        for sample in samples[:2]:
            visualize_sample(trainer, sample, cam_layer_name)

def show_loss_curves(logs):
    from utils.viz import visualize_loss

    visualize_loss(logs)

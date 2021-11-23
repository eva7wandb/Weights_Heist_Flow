from utils import (
    setup, data, viz
)
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
        model_viz=True,
        model_path=None,
        eval_model_on_load=True
    ):
        print(f"[INFO] Loading Data")
        self.train_loader = data.CIFAR10_dataset(
            train=True, cuda=cuda
        ).get_loader()
        self.test_loader = data.CIFAR10_dataset(
            train=False, cuda=cuda
        ).get_loader()
        self.test_loader_unnormalized = data.CIFAR10_dataset(
            train=False, cuda=cuda, normalize=False
        ).get_loader()

        self.net = model.to(device)
        if model_viz:
            viz.show_model_summary(self.net)
        
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=self.lr,
            momentum=0.9, weight_decay=5e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.logs = []
        
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
        print(f"[INFO] Begin training for {EPOCHS} epochs")
        
        for epoch in range(EPOCHS):
            train_batch_loss, train_batch_acc= train(
                self.net, device, 
                self.train_loader, self.optimizer, self.criterion, epoch,
            )
            train_loss = np.mean(train_batch_loss)
            train_acc = np.mean(train_batch_acc)
            test_loss, test_acc = test(
                self.net, device,
                self.test_loader, self.criterion, epoch,
            )
            self.scheduler.step()
            
            ## logging
            log_temp = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "lr": self.optimizer.param_groups[0]['lr'],
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


def show_misclassification(trainer, cam_layer_name='layer4'):
    from utils.viz import visualize_sample
    from utils.testing import get_sample_predictions
    
    sample_preds = get_sample_predictions(trainer)
    
    for class_, samples in sample_preds['mistakes'].items():
        for sample in samples[:2]:
            visualize_sample(trainer, sample, cam_layer_name)
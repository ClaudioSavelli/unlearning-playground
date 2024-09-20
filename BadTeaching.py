import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms

import numpy as np
import random

from Utils.evaluation_metrics import *

import os
import glob
from PIL import Image

from tqdm import tqdm
from math import ceil

#--------------------------------------------------------------------------------
# Bad Teaching 

def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim = 1)
    
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, 
            device, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        # remove second dimension - only for mucac 
        # x = x.squeeze(1)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output = output, labels=y, full_teacher_logits=full_teacher_logits, 
                unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)

def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data, transform, epochs = 1,
                optimizer = 'adam', lr = 0.01, batch_size = 256, device = 'cuda', KL_temperature = 1):
    # creating the unlearning dataset.
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data, transform=transform)
    unlearning_loader = DataLoader(unlearning_data, batch_size = batch_size, shuffle=True)

    print("Number of steps per epoch: ", len(unlearning_loader))

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer#(model.parameters())

    for epoch in range(epochs):
        loss = unlearning_step(model = model, unlearning_teacher= unlearning_teacher, 
                        full_trained_teacher=full_trained_teacher, unlearn_data_loader=unlearning_loader, 
                        optimizer=optimizer, device=device, KL_temperature=KL_temperature)
        print("Epoch {} Unlearning Loss {}".format(epoch+1, loss))

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data, transform):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.transform = transform
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.transform(self.forget_data[index]) if self.transform else self.forget_data[index]
            y = 1
            return x,y
        else:
            x = self.transform(self.retain_data[index - self.forget_len]) if self.transform else self.retain_data[index - self.forget_len]
            y = 0
            return x,y
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model

from torch.utils.data import DataLoader
from sklearn import model_selection

from tqdm import tqdm
import pickle

from Utils.utils import set_seed
import time
import os 

from sklearn.metrics import f1_score

import Utils.CelebA.dataloader_CelebA as dataloader_CelebA
import Utils.MUFAC.dataloader_MUFAC as dataloader_MUFAC
import Utils.MUPins.dataloader_pins as dataloader_pins


@torch.no_grad()
def evaluation(model, data_loader, path, device, save):
    model.eval()
    with torch.no_grad():
        running_corrects = 0

        output_labels = []
        all_labels = []

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # Use the passed model argument here
            _, preds = torch.max(outputs, 1)

            output_labels.append(outputs.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())
        
            running_corrects += torch.sum(preds == labels.data)

        # Use data_loader's dataset length for normalization
        epoch_acc = running_corrects.item() / len(data_loader.dataset)

        output_labels = np.concatenate(output_labels, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        f1 = f1_score(all_labels, np.argmax(output_labels, axis=1), average='weighted')
        
        if save:
            # save in a pickle file
            with open(path + 'output_labels.pkl', 'wb') as f:
                pickle.dump(output_labels, f)
            with open(path + 'labels.pkl', 'wb') as f:
                pickle.dump(all_labels, f)

        return {'Acc': epoch_acc, 'F1': f1}

def compute_losses_binary(net, loader, path, label, device, save):
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    for inputs, labels in loader:
        targets = labels
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)

        losses = criterion(logits, targets).cpu().detach().numpy()
        #losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

        all_logits.append(logits.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
        
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if save:
        with open(path + f'{label}_output_labels.pkl', 'wb') as f:
            pickle.dump(all_logits, f)
        with open(path + f'{label}_labels.pkl', 'wb') as f:
            pickle.dump(all_labels, f)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def cal_mia(model, forget_dataloader_test, test_dataloader, path, device, save):
    set_seed(42)

    forget_losses = compute_losses_binary(model, forget_dataloader_test, path, "forget", device=device, save=save)
    unseen_losses = compute_losses_binary(model, test_dataloader, path, "test", device=device, save=save)

    print(forget_losses.shape, unseen_losses.shape)

    if save: 
        # save in a pickle file
        with open(path + 'forget_losses.pkl', 'wb') as f:
            pickle.dump(forget_losses, f)
        with open(path + 'test_losses.pkl', 'wb') as f:
            pickle.dump(unseen_losses, f)

    if len(forget_losses) > len(unseen_losses):
        np.random.shuffle(forget_losses)
        forget_losses = forget_losses[: len(unseen_losses)]
    elif len(forget_losses) < len(unseen_losses):
        np.random.shuffle(unseen_losses)
        unseen_losses = unseen_losses[: len(forget_losses)]
    
    print(forget_losses.shape, unseen_losses.shape)

    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

    # shuffle the data
    indices = np.arange(len(samples_mia))
    np.random.shuffle(indices)
    samples_mia = samples_mia[indices]
    labels_mia = np.array(labels_mia)[indices]

    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())

    return {'MIA': mia_scores.mean(), 'Forgeting Score': forgetting_score}

def print_evaluation_metrics(model: nn.Module, dataset: str, path: str, save_path: str, device, save = True, use_superclasses_cifar=True):
    
    # val_dataset = dataloader_MUCAC.TestDatasetBinary(source_root=source_root, identities=identities, label_map=label_map, train_index=train_index, transform=test_transform)
    # test_dataset = dataloader_MUCAC.UnseenDatasetBinary(source_root=source_root, identities=identities, label_map=label_map, unseen_index=unseen_index, transform=test_transform)
    # forget_set_test = dataloader_MUCAC.ForgetDatasetBinary(source_root=source_root, identities=identities, label_map=label_map, train_index=train_index, retain_index=retain_index, transform=test_transform)
    
    if dataset == 'celeba':
        val_set = dataloader_CelebA.Dataset_celeba(source_root=path, type='val', transform=dataloader_CelebA.test_transform)
        test_set = dataloader_CelebA.Dataset_celeba(source_root=path, type='test', transform=dataloader_CelebA.test_transform)
        forget_set_test = dataloader_CelebA.Dataset_celeba(source_root=path, type='forget', transform=dataloader_CelebA.test_transform)

    elif dataset == 'mufac':
        val_set = dataloader_MUFAC.get_dataset(label='val', transform=dataloader_MUFAC.test_transform)
        test_set = dataloader_MUFAC.get_dataset(label='test', transform=dataloader_MUFAC.test_transform)
        forget_set_test = dataloader_MUFAC.get_dataset(label='forget', transform=dataloader_MUFAC.test_transform)

    elif dataset == 'pins':
        forget_set_test = dataloader_pins.Dataset_pins('forget', transform=dataloader_pins.test_transform)
        val_set = dataloader_pins.Dataset_pins('val', transform=dataloader_pins.test_transform)
        test_set = dataloader_pins.Dataset_pins('test', transform=dataloader_pins.test_transform)
    else:
        raise ValueError("Dataset not supported")

    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False)
    forget_dataloader_test = torch.utils.data.DataLoader(forget_set_test, batch_size=512, shuffle=False)

    # Performance
    print("Validation on val set") 
    new_path = save_path + "val_acc_outputs/"
    os.makedirs(new_path, exist_ok=True) if save else None
    val_acc = evaluation(model, val_dataloader, new_path, device, save) if dataset != 'mufac' else evaluation_top2(model, val_dataloader, new_path, device, save)
    #print(evaluation_top2(model, val_dataloader, new_path, device, save))
    print("Validation on test set")
    new_path = save_path + "test_acc_outputs/"
    os.makedirs(new_path, exist_ok=True) if save else None
    test_acc = evaluation(model, test_dataloader, new_path, device, save) if dataset != 'mufac' else evaluation_top2(model, test_dataloader, new_path, device, save)
    #print(evaluation_top2(model, test_dataloader, new_path, device, save))
    print("Evaluation of MIA")
    new_path = save_path + "mia_outputs/"
    os.makedirs(new_path, exist_ok=True) if save else None
    mia = cal_mia(model=model, forget_dataloader_test=forget_dataloader_test, test_dataloader=test_dataloader, path=new_path, device=device, save=save)
    print(f'Test Acc: {val_acc}')
    print()
    print(f'Unseen Acc: {test_acc}')
    print()
    print(f'MIA: {mia}')
    print()
    print(f'Final Score: {(val_acc["Acc"] + (1 - abs(mia["MIA"] - 0.5) * 2)) / 2}')
    print()
    dict = {'Test Acc': val_acc, 'Unseen Acc': test_acc, 'MIA': mia, 'Final Score': ((val_acc["Acc"] + (1 - abs(mia["MIA"] - 0.5) * 2)) / 2)}
    return dict
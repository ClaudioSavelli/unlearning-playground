import os
import time
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
import torch.nn.functional as F

from Utils.utils import *

from Utils.dataloader_unlearn import *
from Utils import evaluation_metrics as evaluation_metrics

from Utils.CelebA.dataloader_CelebA import *
import Utils.MUFAC.dataloader_MUFAC as dataloader_MUFAC
from Utils.MUCifar.dataloader_MUCifar import *
import Utils.MUCifar.dataloader_MUCifar as MUCifar
import Utils.MUPins.dataloader_pins as dataloader_pins

num_experiments = 5 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(dataset, model_path, use_superclasses): 
    if dataset == "celeba" or dataset == "pins":
        good_teacher = models.resnet18(weights=None)
        good_teacher.fc = nn.Linear(512, 2)
    elif dataset == "mufac":
        good_teacher = models.resnet18(weights=None)
        num_features = good_teacher.fc.in_features
        good_teacher.fc = nn.Linear(num_features, 8)

    good_teacher.load_state_dict(torch.load(model_path + 'full_model_no_pretrained.pth'))
    good_teacher.to(device)

    # Student initialisation

    if dataset == "celeba" or dataset == "pins":
        student = models.resnet18(weights=None)
        student.fc = nn.Linear(512, 2)
    elif dataset == "mufac":
        student = models.resnet18(weights=None)
        num_features = student.fc.in_features
        student.fc = nn.Linear(num_features, 8)

    student.load_state_dict(torch.load(model_path + 'full_model_no_pretrained.pth'))
    student.to(device)

    # Bad teacher initialisation

    if dataset == "celeba" or dataset == "pins":
        bad_teacher = models.resnet18(weights=None)
        bad_teacher.fc = nn.Linear(512, 2)
    elif dataset == "mufac":
        bad_teacher = models.resnet18(weights=None)
        num_features = bad_teacher.fc.in_features
        bad_teacher.fc = nn.Linear(num_features, 8)

    bad_teacher = bad_teacher.to(device)

    return good_teacher, student, bad_teacher

class Noise(nn.Module):
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

def float_to_uint8(img_float):
    """Convert a floating point image in the range [0,1] to uint8 image in the range [0,255]."""
    img_uint8 = (img_float * 255).astype(np.uint8)
    return img_uint8

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
class SCRUBTraining:
    def __init__(self, teacher, student, retain_dataloader, forget_dataloader, T):
        self.teacher = teacher
        self.student = student
        self.retain_dataloader = retain_dataloader
        self.forget_dataloader = forget_dataloader

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(T)
        self.criterion_kd = DistillKL(T)

        self.optimizer = optim.SGD(student.parameters(), lr=0.001)

    def train_epoch(self):
        self.student.train()
        self.teacher.eval()

        # Function to compute accuracy.
        def compute_accuracy(outputs, labels):
            _, predicted = outputs.max(1)
            total = labels.size(0)
            correct = predicted.eq(labels).sum().item()
            return 100 * correct / total

        total_loss_retain, total_accuracy_retain = 0, 0
        total_loss_forget, total_accuracy_forget = 0, 0

        # Training with retain data.
        for inputs_retain, labels_retain in self.retain_dataloader:
            inputs_retain, labels_retain = inputs_retain.to(device), labels_retain.to(device)

            # Forward pass: Student
            outputs_retain_student = self.student(inputs_retain)

            # Forward pass: Teacher
            with torch.no_grad():
                outputs_retain_teacher = self.teacher(inputs_retain)

            # Loss computation
            loss_cls = self.criterion_cls(outputs_retain_student, labels_retain)
            loss_div_retain = self.criterion_div(outputs_retain_student, outputs_retain_teacher)

            loss = loss_cls + loss_div_retain

            # Update total loss and accuracy for retain data.
            total_loss_retain += loss.item()
            total_accuracy_retain += compute_accuracy(outputs_retain_student, labels_retain)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Training with forget data.
        for inputs_forget, labels_forget in self.forget_dataloader:
            inputs_forget, labels_forget = inputs_forget.to(device), labels_forget.to(device)

            # Forward pass: Student
            outputs_forget_student = self.student(inputs_forget)

            # Forward pass: Teacher
            with torch.no_grad():
                outputs_forget_teacher = self.teacher(inputs_forget)

            # We want to maximize the divergence for the forget data.
            loss_div_forget = -self.criterion_div(outputs_forget_student, outputs_forget_teacher)

            # Update total loss and accuracy for forget data.
            total_loss_forget += loss_div_forget.item()
            total_accuracy_forget += compute_accuracy(outputs_forget_student, labels_forget)

            # Backward pass
            self.optimizer.zero_grad()
            loss_div_forget.backward()
            self.optimizer.step()

        # Print average loss and accuracy for the entire epoch
        avg_loss_retain = total_loss_retain / len(self.retain_dataloader)
        avg_accuracy_retain = total_accuracy_retain / len(self.retain_dataloader)

        avg_loss_forget = total_loss_forget / len(self.forget_dataloader)
        avg_accuracy_forget = total_accuracy_forget / len(self.forget_dataloader)

        print(f'Epoch Retain: Avg Loss: {avg_loss_retain:.4f}, Avg Accuracy: {avg_accuracy_retain:.2f}%')
        print(f'Epoch Forget: Avg Loss: {avg_loss_forget:.4f}, Avg Accuracy: {avg_accuracy_forget:.2f}%')
    
if __name__ == "__main__":
    dataset = "celeba"
    use_superclasses = False

    if dataset == "celeba":
        path = "SBS/Datasets/CelebA_Unlearning/"
    elif dataset == "mufac":
        path = "SBS/Datasets/MUFAC/"
    elif dataset == "pins":
        path = "SBS/Datasets/Pins/"
    else:
        raise ValueError("Dataset not recognized")

    if dataset == "celeba" or dataset == "mufac":
        train_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor()
        ])
    elif dataset == "pins":
        train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
        
    if dataset == "celeba": 
        train_set = Dataset_celeba(source_root=path, type="train", transform=train_transform)
        val_set = Dataset_celeba(source_root=path, type="val", transform=test_transform)
        forget_set_train = Dataset_celeba(source_root=path, type="forget", transform=train_transform)
        forget_set_test = Dataset_celeba(source_root=path, type="forget", transform=test_transform)
        retain_set_train = Dataset_celeba(source_root=path, type="retain", transform=train_transform)
        retain_set_test = Dataset_celeba(source_root=path, type="retain", transform=test_transform)
        test_set = Dataset_celeba(source_root=path, type="test", transform=test_transform)
        
    elif dataset == "mufac":
        train_meta_data_path = 'SBS/Datasets/MUFAC/unique_train_dataset.csv'
        train_meta_data = pd.read_csv(train_meta_data_path)
        train_image_directory = 'SBS/Datasets/MUFAC/unique_train_images'
        train_set = dataloader_MUFAC.Dataset_MUFAC(meta_data = train_meta_data, image_directory = train_image_directory, transform=test_transform)

        val_meta_data_path = 'SBS/Datasets/MUFAC/unique_test_dataset.csv'
        val_meta_data = pd.read_csv(val_meta_data_path)
        val_image_directory = 'SBS/Datasets/MUFAC/unique_test_images'
        val_set = dataloader_MUFAC.Dataset_MUFAC(meta_data = val_meta_data, image_directory = val_image_directory, transform=test_transform)

        forget_set_train = dataloader_MUFAC.get_dataset(label='forget', transform=train_transform) # to use for sbs -> test_transform
        retain_set_train = dataloader_MUFAC.get_dataset(label='retain', transform=train_transform)
        forget_set_test = dataloader_MUFAC.get_dataset(label='forget', transform=test_transform) # to use for sbs -> test_transform
        retain_set_test = dataloader_MUFAC.get_dataset(label='retain', transform=test_transform) # to use for sbs -> test_transform

    elif dataset == "pins":
        train_set = dataloader_pins.Dataset_pins(split='train', transform=train_transform)
        retain_set_train = dataloader_pins.Dataset_pins(split='retain', transform=train_transform)
        retain_set_test = dataloader_pins.Dataset_pins(split='retain', transform=test_transform)
        forget_set_train = dataloader_pins.Dataset_pins(split='forget', transform=train_transform)
        forget_set_test = dataloader_pins.Dataset_pins(split='forget', transform=test_transform)
        val_set = dataloader_pins.Dataset_pins(split='val', transform=test_transform)
        test_set = dataloader_pins.Dataset_pins(split='test', transform=test_transform)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
    forget_dataloader_train = torch.utils.data.DataLoader(forget_set_train, batch_size=64, shuffle=True)
    forget_dataloader_test = torch.utils.data.DataLoader(forget_set_test, batch_size=2048, shuffle=False)
    retain_dataloader_train = torch.utils.data.DataLoader(retain_set_train, batch_size=64, shuffle=True)
    retain_dataloader_test = torch.utils.data.DataLoader(retain_set_test, batch_size=2048, shuffle=False)

    if dataset == "celeba":
        model_path = "SBS/Models/Thesis/celeba/"
    elif dataset == "mufac":
        model_path = "SBS/Models/Thesis/mufac/"
    elif dataset == "pins":
        model_path = "SBS/Models/Thesis/pins/"

    save_path_base = f"thesis_results/other_methods_{dataset}_temp" + ("_superclasses" if use_superclasses else "") + "/" 
    print("Save path base: ", save_path_base)
    os.makedirs(save_path_base, exist_ok=True)

    # fine-tuning 

    for ne in range(num_experiments):
        for num_epochs in [1, 2, 3]:
            save_path = save_path_base + f'fine-tuning_nepochs{num_epochs}/{ne}/'
            os.makedirs(save_path, exist_ok=True)
            
            unlearned_model, _, _ = load_models(dataset, model_path, use_superclasses)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

            sum_of_times = 0

            tic = time.time()
            set_seed(ne)
            for epoch in range(num_epochs):
                running_loss = 0

                for batch_idx, (x_retain, y_retain) in enumerate(retain_dataloader_train):
                    y_retain = y_retain.to(device)

                    # Classification Loss
                    outputs_retain = unlearned_model(x_retain.to(device))
                    classification_loss = criterion(outputs_retain, y_retain)

                    optimizer.zero_grad()
                    classification_loss.backward()
                    optimizer.step()

                    running_loss += classification_loss.item() * x_retain.size(0)
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {classification_loss.item():.4f}")

                average_epoch_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
                print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")
            
            total_time = time.time() - tic

            res = evaluation_metrics.print_evaluation_metrics(model=unlearned_model, 
                                                                dataset=dataset, 
                                                                path=path, 
                                                                save_path=save_path, 
                                                                device=device, 
                                                                save=True,
                                                                use_superclasses_cifar=use_superclasses)

            # save experiments results in a pickle file 
            with open(save_path + f"{ne}.txt", 'w') as f:
                for key, value in res.items():
                    f.write('%s:%s\n' % (key, value))
                f.write('Time:%s\n' % (total_time))
    
    # CF-k
    for ne in range(num_experiments):
        for freezed_layers in [1, 2, 3]:
            for num_epochs in [1, 2, 3]:
                save_path = save_path_base + f'CF-{freezed_layers}_nepochs{num_epochs}/{ne}/'
                os.makedirs(save_path, exist_ok=True)
                
                unlearned_model, _, _ = load_models(dataset, model_path, use_superclasses)
                
                criterion = nn.CrossEntropyLoss()

                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, unlearned_model.parameters()), lr=0.001)

                # Freeze all the parameters.
                for param in unlearned_model.parameters():
                    param.requires_grad = False

                # Only unfreeze the last three layers for the fine-tuning.
                if freezed_layers >= 3:
                    for param in unlearned_model.layer3.parameters():
                        param.requires_grad = True
                if freezed_layers >= 2:
                    for param in unlearned_model.layer4.parameters():
                        param.requires_grad = True
                if freezed_layers >= 1:
                    for param in unlearned_model.avgpool.parameters():
                        param.requires_grad = True
                    for param in unlearned_model.fc.parameters():
                        param.requires_grad = True

                tic = time.time()
                set_seed(ne)
                for epoch in range(num_epochs):
                    running_loss = 0

                    for batch_idx, (x_retain, y_retain) in enumerate(retain_dataloader_train):
                        y_retain = y_retain.to(device)

                        # Classification Loss
                        outputs_retain = unlearned_model(x_retain.to(device))
                        classification_loss = criterion(outputs_retain, y_retain)

                        optimizer.zero_grad()
                        classification_loss.backward()
                        optimizer.step()

                        running_loss += classification_loss.item() * x_retain.size(0)
                        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {classification_loss.item():.4f}")

                    average_epoch_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
                    print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")
                
                total_time = time.time() - tic

                res = evaluation_metrics.print_evaluation_metrics(model=unlearned_model, 
                                                                    dataset=dataset, 
                                                                    path=path, 
                                                                    save_path=save_path, 
                                                                    device=device, 
                                                                    save=True,
                                                                    use_superclasses_cifar=use_superclasses)

                # save experiments results in a pickle file
                with open(save_path + f"{ne}.txt", 'w') as f:
                    for key, value in res.items():
                        f.write('%s:%s\n' % (key, value))
                    f.write('Time:%s\n' % (total_time))
    
    # NegGrad
    for ne in range(num_experiments):
        for num_epochs in [1, 2, 3]:
            save_path = save_path_base + f'NormalNegGrad_nepochs_{num_epochs}/{ne}/'
            os.makedirs(save_path, exist_ok=True)
            
            unlearned_model, _, _ = load_models(dataset, model_path, use_superclasses)
            
            criterion = nn.CrossEntropyLoss()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

            dataloader_iterator = iter(forget_dataloader_train)

            tic = time.time()
            set_seed(ne)
            for epoch in range(num_epochs):
                running_loss = 0

                for batch_idx, (x_retain, y_retain) in enumerate(retain_dataloader_train):
                    y_retain = y_retain.to(device)

                    try:
                        (x_forget, y_forget) = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(forget_dataloader_train)
                        (x_forget, y_forget) = next(dataloader_iterator)

                    if x_forget.size(0) != x_retain.size(0):
                        continue

                    outputs_forget = unlearned_model(x_forget.to(device))
                    loss_ascent_forget = -criterion(outputs_forget, y_forget.to(device))

                    optimizer.zero_grad()
                    loss_ascent_forget.backward()
                    optimizer.step()

                    running_loss += loss_ascent_forget.item() * x_retain.size(0)
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {loss_ascent_forget.item():.4f}")

                average_epoch_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
                print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")
            
            total_time = time.time() - tic

            res = evaluation_metrics.print_evaluation_metrics(model=unlearned_model, 
                                                                dataset=dataset, 
                                                                path=path, 
                                                                save_path=save_path, 
                                                                device=device, 
                                                                save=True,
                                                                use_superclasses_cifar=use_superclasses)

            with open(save_path + f"{ne}.txt", 'w') as f:
                for key, value in res.items():
                    f.write('%s:%s\n' % (key, value))
                f.write('Time:%s\n' % (total_time))

    # Advanced NegGrad
    for ne in range(num_experiments):
        for num_epochs in [1, 2, 3]:    
            save_path = save_path_base + f'AdvancedNegGrad_nepochs_{num_epochs}/{ne}/'  
            os.makedirs(save_path, exist_ok=True)
            
            unlearned_model, _, _ = load_models(dataset, model_path, use_superclasses)
            
            criterion = nn.CrossEntropyLoss()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

            dataloader_iterator = iter(forget_dataloader_train)

            tic = time.time()
            set_seed(ne)
            for epoch in range(num_epochs):
                running_loss = 0

                for batch_idx, (x_retain, y_retain) in enumerate(retain_dataloader_train):
                    y_retain = y_retain.to(device)

                    try:
                        (x_forget, y_forget) = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(forget_dataloader_train)
                        (x_forget, y_forget) = next(dataloader_iterator)

                    if x_forget.size(0) != x_retain.size(0):
                        continue

                    outputs_retain = unlearned_model(x_retain.to(device))
                    outputs_forget = unlearned_model(x_forget.to(device))

                    loss_ascent_forget = -criterion(outputs_forget, y_forget.to(device))
                    loss_retain = criterion(outputs_retain, y_retain.to(device))

                    # Overall loss
                    joint_loss = loss_ascent_forget + loss_retain

                    print("joint loss :", joint_loss.item())
                    optimizer.zero_grad()
                    joint_loss.backward()
                    optimizer.step()

                    running_loss += joint_loss.item() * x_retain.size(0)
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {joint_loss.item():.4f}")

                average_epoch_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
                print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")
            
            total_time = time.time() - tic

            res = evaluation_metrics.print_evaluation_metrics(model=unlearned_model, 
                                                                dataset=dataset, 
                                                                path=path, 
                                                                save_path=save_path, 
                                                                device=device, 
                                                                save=True,
                                                                use_superclasses_cifar=use_superclasses)

            with open(save_path + f"{ne}.txt", 'w') as f:
                for key, value in res.items():
                    f.write('%s:%s\n' % (key, value))
                f.write('Time:%s\n' % (total_time))

    # UNSIR-1

    for ne in range(5):
        for num_epochs_a in [1, 2, 3]: 
            save_path = save_path_base + f'UNSIR_1_nepochs{num_epochs_a}/{ne}/'
            
            unlearned_model, _, _ = load_models(dataset, model_path, use_superclasses)
            
            criterion = nn.CrossEntropyLoss()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

            print_interval = 1
            train_epoch_losses = []

            tic = time.time()
            set_seed(ne)
            for epoch in range(num_epochs_a):
                running_loss = 0

                for batch_idx, ((x_retain, y_retain), (x_forget, y_forget)) in enumerate(zip(retain_dataloader_train, forget_dataloader_train)):
                    y_retain = y_retain.to(device)
                    batch_size_forget = y_forget.size(0)

                    if x_retain.size(0) != 64 or x_forget.size(0) != 64:
                        continue

                    # Initialize the noise.
                    noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
                    noise = Noise(batch_size_forget, *noise_dim).to(device)
                    noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.01)
                    noise_tensor = noise()[:batch_size_forget]

                    # Update the noise for increasing the loss value.
                    for _ in range(5):
                        outputs = unlearned_model(noise_tensor)
                        with torch.no_grad():
                            target_logits = unlearned_model(x_forget.to(device))
                        # Maximize the similarity between noise data and forget features.
                        loss_noise = -F.mse_loss(outputs, target_logits)

                        # Backpropagate to update the noise.
                        noise_optimizer.zero_grad()
                        loss_noise.backward(retain_graph=True)
                        noise_optimizer.step()

                    # Train the model with noise and retain image
                    noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().to(device)
                    outputs = unlearned_model(noise_tensor.to(device))
                    loss_1 = criterion(outputs, y_retain)

                    outputs = unlearned_model(x_retain.to(device))
                    loss_2 = criterion(outputs, y_retain)

                    joint_loss = loss_1 + loss_2

                    optimizer.zero_grad()
                    joint_loss.backward()
                    optimizer.step()
                    running_loss += joint_loss.item() * x_retain.size(0)

                    if batch_idx % print_interval == 0:
                        print(f"Epoch [{epoch+1}/{num_epochs_a}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {joint_loss.item():.4f}")

                average_train_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
                train_epoch_losses.append(average_train_loss)
                print(f"Epoch [{epoch+1}/{num_epochs_a}] - Train Loss: {average_train_loss:.4f}")

            total_time = time.time() - tic

            res = evaluation_metrics.print_evaluation_metrics(model=unlearned_model, 
                                                                dataset=dataset, 
                                                                path=path, 
                                                                save_path=save_path, 
                                                                device=device, 
                                                                save=True,
                                                                use_superclasses_cifar=use_superclasses)

            with open(save_path + f"{ne}.txt", 'w') as f:
                for key, value in res.items():
                    f.write('%s:%s\n' % (key, value))
                f.write('Time:%s\n' % (total_time))

    # UNSIR-2

    for ne in range(5):
        for num_epochs_a in [1, 2, 3]: 
            save_path = save_path_base + f'UNSIR_1_nepochs{num_epochs_a}/{ne}/'
            
            unlearned_model, _, _ = load_models(dataset, model_path, use_superclasses)
            
            criterion = nn.CrossEntropyLoss()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

            print_interval = 1
            train_epoch_losses = []

            tic = time.time()
            set_seed(ne)
            for epoch in range(num_epochs_a):
                running_loss = 0

                for batch_idx, ((x_retain, y_retain), (x_forget, y_forget)) in enumerate(zip(retain_dataloader_train, forget_dataloader_train)):
                    y_retain = y_retain.to(device)
                    batch_size_forget = y_forget.size(0)

                    if x_retain.size(0) != 64 or x_forget.size(0) != 64:
                        continue

                    # Initialize the noise.
                    noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
                    noise = Noise(batch_size_forget, *noise_dim).to(device)
                    noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.01)
                    noise_tensor = noise()[:batch_size_forget]

                    # Update the noise for increasing the loss value.
                    for _ in range(5):
                        outputs = unlearned_model(noise_tensor)
                        with torch.no_grad():
                            target_logits = unlearned_model(x_forget.to(device))
                        # Maximize the similarity between noise data and forget features.
                        loss_noise = -F.mse_loss(outputs, target_logits)

                        # Backpropagate to update the noise.
                        noise_optimizer.zero_grad()
                        loss_noise.backward(retain_graph=True)
                        noise_optimizer.step()

                    # Train the model with noise and retain image
                    noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().to(device)
                    outputs = unlearned_model(noise_tensor.to(device))
                    loss_1 = criterion(outputs, y_retain)

                    outputs = unlearned_model(x_retain.to(device))
                    loss_2 = criterion(outputs, y_retain)

                    joint_loss = loss_1 + loss_2

                    optimizer.zero_grad()
                    joint_loss.backward()
                    optimizer.step()
                    running_loss += joint_loss.item() * x_retain.size(0)

                    if batch_idx % print_interval == 0:
                        print(f"Epoch [{epoch+1}/{num_epochs_a}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {joint_loss.item():.4f}")

                average_train_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
                train_epoch_losses.append(average_train_loss)
                print(f"Epoch [{epoch+1}/{num_epochs_a}] - Train Loss: {average_train_loss:.4f}")

            for num_epochs_b in [1, 2, 3]:
                save_path = save_path_base + f'UNSIR_2_nepocsa_{num_epochs_a}_nepocsb_{num_epochs_b}/{ne}/'

                criterion = torch.nn.CrossEntropyLoss()
                optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

                set_seed(ne)
                for epoch in range(num_epochs_b):
                    running_loss = 0

                    for batch_idx, (x_retain, y_retain) in enumerate(retain_dataloader_train):
                        y_retain = y_retain.to(device)

                        # Classification Loss
                        outputs_retain = unlearned_model(x_retain.to(device))
                        classification_loss = criterion(outputs_retain, y_retain)

                        optimizer.zero_grad()
                        classification_loss.backward()
                        optimizer.step()

                        running_loss += classification_loss.item() * x_retain.size(0)
                        print(f"Epoch [{epoch+1}/{num_epochs_b}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {classification_loss.item():.4f}")

                    average_epoch_loss = running_loss / (len(retain_dataloader_train) * x_retain.size(0))
                    print(f"Epoch [{epoch+1}/{num_epochs_b}] - Total Loss: {running_loss:.4f}")
                
            total_time = time.time() - tic

            res = evaluation_metrics.print_evaluation_metrics(model=unlearned_model, 
                                                                dataset=dataset, 
                                                                path=path, 
                                                                save_path=save_path, 
                                                                device=device, 
                                                                save=True,
                                                                use_superclasses_cifar=use_superclasses)

            with open(save_path + f"{ne}.txt", 'w') as f:
                for key, value in res.items():
                    f.write('%s:%s\n' % (key, value))
                f.write('Time:%s\n' % (total_time))

    # SCRUB
    
    for ne in range(num_experiments):
        for num_epochs in [1, 2, 3]:
            for T in [3.0, 4.0, 5.0]: 
                save_path = save_path_base + f'SCRUB_T_{T}_nepochs_{num_epochs}/{ne}/'
                if not os.path.exists(save_path):

                    teacher, student, _ = load_models(dataset, model_path, use_superclasses)
                    
                    criterion = nn.CrossEntropyLoss()

                    # Initialize and train
                    scrub_trainer = SCRUBTraining(teacher, student, retain_dataloader_train, forget_dataloader_train, T)

                    tic = time.time()
                    set_seed(ne)
                    for epoch in range(num_epochs):
                        scrub_trainer.train_epoch()
                        print(f"Epoch {epoch+1} completed.")
                    
                    total_time = time.time() - tic

                    res = evaluation_metrics.print_evaluation_metrics(model=student, 
                                                                        dataset=dataset, 
                                                                        path=path, 
                                                                        save_path=save_path, 
                                                                        device=device, 
                                                                        save=True,
                                                                        use_superclasses_cifar=use_superclasses)

                    with open(save_path + f"{ne}.txt", 'w') as f:
                        for key, value in res.items():
                            f.write('%s:%s\n' % (key, value))
                        f.write('Time:%s\n' % (total_time))
                else:
                    print(f"Path {save_path} already exists.")
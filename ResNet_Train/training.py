import time
import datetime
import copy
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
from torch.utils.data import random_split, ConcatDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np

def prepare_data(batch_size=128, num_workers=2, split=0.5):
    # Load CIFAR-100 dataset
    initial_transforms = transforms.ToTensor()
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=initial_transforms)
    val_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=initial_transforms)

    # Combine train and test datasets
    full_dataset = ConcatDataset([train_dataset, val_dataset])

    # Split into new train and val sets
    train_size = int((1 - split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # CIFAR-100 normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define transforms with the calculated mean and std
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Apply transforms to the respective sets
    train_set.dataset.transform = train_transforms
    val_set.dataset.transform = val_transforms

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader}

def train(model, dataloaders, loss_fn, optimizer, num_epochs, device):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time}")

    for epoch in range(num_epochs):
        print(f'##Epoch {epoch + 1}##')

        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            loss_epoch = 0.0
            corrects_epoch = 0

            for x, y in dataloaders[mode]:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    _, preds = torch.max(outputs, 1)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                loss_epoch += loss.item() * x.size(0)
                corrects_epoch += torch.sum(preds == y.data)

            epoch_loss = loss_epoch / len(dataloaders[mode].dataset)
            epoch_acc = corrects_epoch / len(dataloaders[mode].dataset)

            print(f'{mode} Loss: {epoch_loss:.5f} Accuracy: {epoch_acc:.5f}')

            if mode == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    print(f"Training completed at: {end_time}")
    print(f"Total training time: {elapsed_time}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    model.load_state_dict(best_model_wts)
    return model

def export_softmax_scores(model, dataloader, device, output_path):
    model.eval()
    softmax_fn = nn.Softmax(dim=1)
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            scores = softmax_fn(outputs).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save(output_path + 'scores.npy', all_scores)
    np.save(output_path + 'labels.npy', all_labels)
    print(f"Softmax scores and labels saved: {output_path}scores.npy and {output_path}labels.npy")


n_classes = 100
learning_rate = 1e-4
n_epochs = 30
model_name = 'cifar100_model'
output_path = './results/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

dataloaders = prepare_data()

model = resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = train(model, dataloaders, loss_func, optimizer, n_epochs, device)

torch.save(model.state_dict(), f'{output_path}{model_name}.pth')
print(f'Saved: {output_path}{model}.pth')

export_softmax_scores(model, dataloaders['val'], device, output_path)
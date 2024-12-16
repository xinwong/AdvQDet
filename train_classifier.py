import os
import argparse
import random
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

def set_seed(args):
    """
    :param args:
    :return:
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

def main(args):

    # set seed
    set_seed(args)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    if args.data == "gtsrb":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.GTSRB(root='./data', split="train", transform=transform, download=True)
        val_dataset = datasets.GTSRB(root='./data', split="test", transform=transform, download=True)
    elif args.data == "flowers102":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.Flowers102(root='./data', split="train", transform=transform, download=True)
        val_dataset = datasets.Flowers102(root='./data', split="val", transform=transform, download=True)
        test_dataset = datasets.Flowers102(root='./data', split="test", transform=transform, download=True)
    elif args.data == "oxford-iiit-pet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.OxfordIIITPet(root='./data', split="trainval", target_types="category", transform=transform, download=True)
        val_dataset = datasets.OxfordIIITPet(root='./data', split="test", target_types="category", transform=transform, download=True)        
    else:
        raise ValueError("Dataset not supported.")
    
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=8)

    # model
    if args.data == "gtsrb":
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 43)
    elif args.data == "flowers102":
        # # Load ViT model pre-trained on ImageNet
        # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=102)
        model = models.resnet101(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 102)
        model = model.to(device)
    elif args.data == "oxford-iiit-pet":
        # # Load ViT model pre-trained on ImageNet
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=37)
        model = model.to(device)
    else:
        raise ValueError("data name error!")
    
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    best_accuracy = 0.0
    num_epochs = args.epoch

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy on validation set: {accuracy:.4f}')

        # check if model performance improved or not, if yes, save the model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # save best model
            torch.save(model, 'vit_{}.pth'.format(args.data))

    if args.test:      
        print("val acc: ",best_accuracy)
        test(args, test_dataset, model, device)

def test(args, test_dataset, model, device):
    # Load the test dataset
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False, num_workers=8)

    # Test the model
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

        print(f'Test Accuracy: {float(num_correct)/float(num_samples)*100:.2f}%')

    print("Testing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-b', '--batch', default=64, type=int)
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-d', '--data', default="oxford-iiit-pet", type=str)    # oxford-iiit-pet, flowers102, gtsrb
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
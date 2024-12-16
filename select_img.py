import argparse
import json
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageNet, GTSRB, Flowers102, OxfordIIITPet
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights

def get_correctly_classified_images(model, data_loader, device, args):
    correctly_classified_images = {}

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if args.data == "imagenet" or args.data == "flowers-102":
                xform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                images_x = xform(images)
            outputs = model(images_x)
            _, predicted = torch.max(outputs, 1)
            correct_mask = predicted == labels

            for i in range(len(labels)):
                label = labels[i].item()
                if label not in correctly_classified_images and correct_mask[i].item():
                    correctly_classified_images[label] = images[i]

    return correctly_classified_images

def get_10_correctly_classified_images(model, data_loader, device, class_num=10):
    # Initialize a dictionary with keys as labels and values as empty lists
    correctly_classified_images = {label: [] for label in range(class_num)}

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if args.data == "oxford-iiit-pet":
                xform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                images_x = xform(images)
            else:
                images_x = images
            outputs = model(images_x)
            _, predicted = torch.max(outputs, 1)
            correct_mask = predicted == labels

            for i in range(len(labels)):
                label = labels[i].item()
                # Check if the image is correctly classified and less than 10 images are stored for the label
                if correct_mask[i].item() and len(correctly_classified_images[label]) < 3:
                    correctly_classified_images[label].append(images[i])

                # Break if all labels have 10 images
                if all(len(images) == 3 for images in correctly_classified_images.values()):
                    return correctly_classified_images

    return correctly_classified_images



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
    set_seed(args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == "cifar10":
        # Load pre-trained model (you should replace 'your_model.pth' with the actual model file path)
        model = torch.load("models/pretrained/resnet20-12fca82f-single.pth", map_location="cpu")
        model = model.to(device)

        # Load CIFAR-10 test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = CIFAR10(root='../SimilarityDetection/data', train=False, download=True, transform=transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    elif args.data == "imagenet":
        # Load pre-trained model (you should replace 'your_model.pth' with the actual model file path)
        model = resnet152(pretrained=True)
        model = model.to(device)

        # Load ImageNet dataset
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        dataset = ImageNet(root='../SimilarityDetection/data/imagenet', split="val", transform=transform)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
    elif args.data == "gtsrb":
        model = torch.load("./models/pretrained/resnet34_gtsrb.pth")
        model = model.to(device)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        dataset = GTSRB(root='./data', split="test", transform=transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    elif args.data == "flowers-102":
        model = torch.load("./models/pretrained/resnet101_flowers102.pth")
        model = model.to(device)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        dataset = Flowers102(root='./data', split="test", transform=transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    elif args.data == "oxford-iiit-pet":
        model = torch.load("./models/pretrained/vit_oxford-iiit-pet.pth")
        model = model.to(device)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        dataset = OxfordIIITPet(root='./data', split="test", target_types="category", transform=transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    else:
        raise ValueError("Dataset not supported.")

    # Get correctly classified images for each class
    if args.data == "cifar10":
        correctly_classified_images = get_10_correctly_classified_images(model, data_loader, device, class_num=10)
    elif args.data == "gtsrb":
        correctly_classified_images = get_10_correctly_classified_images(model, data_loader, device, class_num=43)
    elif args.data == "flowers-102":
        correctly_classified_images = get_correctly_classified_images(model, data_loader, device, args)
    elif args.data == "oxford-iiit-pet":
        correctly_classified_images = get_10_correctly_classified_images(model, data_loader, device, class_num=37)
    else:
        correctly_classified_images = get_correctly_classified_images(model, data_loader, device, args)

    # Save or display the selected images
    if args.data == "cifar10" or args.data == "gtsrb" or args.data == "oxford-iiit-pet":
        # Save or display the selected images
        for label, images in correctly_classified_images.items():
            for idx, image in enumerate(images):
                _, pre = torch.max(model(image.unsqueeze(0).to(device)), 1)
                print(f'Label: {label}, Prediction: {pre.item()}')

                # Save the image to a folder
                image_file = f'./data/{args.data}/imgs/{label*3+idx}.png'
                torchvision.utils.save_image(image, image_file)

        # Generate a dictionary for JSON
        json_data = {f'imgs/{label*3+idx}.png': label for label, images in correctly_classified_images.items() for idx in range(len(images))}
        json_file_path = './data/{}/{}.json'.format(args.data, args.data)

        # Write the dictionary to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
    else:
        for label, image in correctly_classified_images.items():
            _, pre = torch.max(model(image.unsqueeze(0)), 1)
            print(label, pre)

            # Generate a dictionary
            json_data = {f'imgs/{i}.png': i for i in range(len(correctly_classified_images))}
            json_file_path = './data/{}/{}.json'.format(args.data, args.data)
            # Write the dictionary to a JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

            # Save the images to a folder
            torchvision.utils.save_image(image, f'./data/{args.data}/imgs/{label}.png')
            # Alternatively, you can display the images using a plotting library like matplotlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='set seed for model', default=1, type=int)
    parser.add_argument('-d', '--data', help='cifar or imagenet', default="oxford-iiit-pet", type=str)
    args = parser.parse_args()
    main(args)
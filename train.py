
# Udacity AI Programming in Python Nanodegree - Final project
#
# Author: Perry Brandiezs
#
# train.py - this program will train the network.
#
# Usage: train.py -h
#
# Example: python train.py --save_dir save --gpu flowers
#


import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
import torchvision


# Get command line arguments
parser = argparse.ArgumentParser(description='Train the network')
parser.add_argument('data_directory', type=str, help='data_directory')
parser.add_argument('--save_dir', '-s', type=str, dest='save_directory', default='save', help='Set directory to save checkpoints')
parser.add_argument('--arch', dest='architecture', default='vgg11', help='Set the architecture -vgg11 or vgg13 are valid choices, default vgg11')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.001, help='Set the learning rate, default 0.001')
parser.add_argument('--hidden_units', type=int, dest='hidden_units', default=512, help='Set the hidden units, default 512')
parser.add_argument('--epochs', type=int, dest='epochs', default=3, help='Set the number of epochs, default 3')
parser.add_argument('--gpu', action="store_true", dest='gpu', default=False, help='Use gpu for training, default False')
args = parser.parse_args()
print(args)

print("")
print("epochs", args.epochs)
epochs = args.epochs

# set the model to use
# model = "models." + args.architecture + "(pretrained=True)"
# restrict model to vgg11 and vgg13
allowed_models=['vgg11', 'vgg13']
if args.architecture not in allowed_models:
    print("Allowed models are vgg11 or vgg13")
    exit(1)
arch = args.architecture
model = models.__dict__[args.architecture](pretrained=True)

# set the data_dir
data_dir = args.data_directory
print("Data Directory is:", data_dir)

# set the save_directory
save_directory = args.save_directory
print("Save Directory is:", save_directory)
save_target = save_directory + "/checkpoint.pth"
print("Save Target is:", save_target)

# set the learning rate
learning_rate = args.learning_rate
print("Learning Rate is:", learning_rate)

# set the hidden units
hidden_units = args.hidden_units
print("Hidden units are:", hidden_units)

# set the gpu use (boolean), default True
gpu = args.gpu
print("GPU is:", gpu)
if not torch.cuda.is_available():
    print("GPU not found -> setting gpu to False")
    gpu = False

# set the train_dir, valid_dir, and test_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print("train_dir:", train_dir)
print("valid_dir:", valid_dir)
print("test_dir:", test_dir)


# set the transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
                                                           
data_transforms = {'train':train_transforms, 'test':test_transforms, 'validation':validation_transforms}

# Load the datasets with ImageFolder

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

image_datasets = {'train':train_data, 'test':test_data, 'validation':validation_data}

# Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)

dataloaders = {'train':trainloader, 'test':testloader, 'validation':validationloader}

# Get the names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Build and train your network

# freeze
for param in model.parameters():
    param.requires_grad = False

# set the classifier
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
print(model)

#Train
print("Training started..")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
print_every = 40
steps = 0

# change to cuda
if gpu:
    model.to('cuda')
else:
    model.to('cpu')


for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        if gpu:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        else:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0

            # Do validation on the validation set
            correct = 0
            total = 0
            with torch.no_grad():
                for validation_data in validationloader:
                    validation_images, validation_labels = validation_data
                    if gpu:
                        validation_images = validation_images.to('cuda')
                        validation_labels = validation_labels.to('cuda')

                    validation_outputs = model(validation_images)
                    _, predicted = torch.max(validation_outputs.data, 1)
                    total += validation_labels.size(0)
                    correct += (predicted == validation_labels).sum().item()
            print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))


print("Training finished..")




# Test images

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if gpu:
            images = images.to('cuda')
            labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
print("Validation finished..")

# Save the checkpoint

print("Saving checkpoint..")

model.cpu
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'epochs': epochs,
              'loss': loss,
              'learning_rate': learning_rate,
              'architecture': arch,
              'model_state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'model_classifier': classifier,
              'class_to_idx': model.class_to_idx,
              }
torch.save(checkpoint, save_target)


print("Checkpoint saved..")
# Udacity AI Programming in Python Nanodegree - Final project
#
# Author: Perry Brandiezs
#
# predict.py - this program will be used to predict a classification from an image
#
# Usage: predict.py -h
#
# Example: python predict.py --gpu flowers/test/1/image_06743.jpg save
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
parser = argparse.ArgumentParser(description='Use a trained network to predict the class for an input image')
parser.add_argument('input', type=str, help='Path to input image')
parser.add_argument('checkpoint', type=str, help='Path to checkpoint save directory')
parser.add_argument('--top_k', type=int, dest='top_k', default=3, help='top_k values to display, default 3')
parser.add_argument('--gpu', action="store_true", dest='gpu', default=False, help='Use gpu for inference, default True')
parser.add_argument('--mapping', type=str, dest='mapping', default='cat_to_name.json', help='Path to json mapping file, default cat_to_name.json')
# parser.add_argument('--arch', dest='architecture', default='vgg11', help='Set the architecture -vgg11 or vgg13 are valid choices, default vgg11')
# parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.001, help='Set the learning rate, default 0.001')
# parser.add_argument('--hidden_units', type=int, dest='hidden_units', default=512, help='Set the hidden units, default 512')
args = parser.parse_args()
print(args)

print()
input_image = args.input
print("Input image is:", input_image)

checkpoint = args.checkpoint + "/checkpoint.pth"
print("Checkpoint location is", checkpoint)

gpu = args.gpu
print("GPU setting is:",gpu)
if not torch.cuda.is_available():
    print("GPU not found -> setting gpu to False")
    gpu = False
    

top_k = args.top_k
print("top_k setting is:", top_k)

mapping = args.mapping
print("JSON mapping file is:", mapping)

# restrict model to vgg11 and vgg13
# allowed_models=['vgg11', 'vgg13']
# if args.architecture not in allowed_models:
#     print("Allowed models are vgg11 or vgg13")
#     exit(1)
# model = models.__dict__[args.architecture](pretrained=True)

# Define settings
# learning_rate = args.learning_rate
# hidden_units = args.hidden_units

# get the names
with open(mapping, 'r') as f:
    cat_to_name = json.load(f)



# Load previously saved checkpoint

# Loads checkpoint and rebuilds the model
def load_checkpoint(filepath):
    if not gpu:
        checkpoint = torch.load(filepath, map_location={'cuda:0': 'cpu'})
    else:
        checkpoint = torch.load(filepath)
    classifier = checkpoint['model_classifier']
    model = getattr(torchvision.models, checkpoint['architecture'])(pretrained=True)
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    model.loss = checkpoint['loss']
    model.learning_rate = checkpoint['learning_rate']
    model.arch = checkpoint['architecture']
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return model



model = load_checkpoint(checkpoint)
model.eval()
# print(model)

# freeze
for param in model.parameters():
    param.requires_grad = False

# optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)





#Process the test image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #open the image
    im = Image.open(image)

    #resize the image
    # im = im.resize((255,255))
    size = 256
    width = im.width
    height = im.height
    # print("width", width)
    # print("height", height)
    shortest_side = min(width, height)
    # print("shortest_side", shortest_side)
    # print("size", size)
    # print("width / shortest_side * size", width / shortest_side * size)
    # print("height / shortest_side * size", height / shortest_side * size)
    im = im.resize((int((width/shortest_side)*size), int((height/shortest_side)*size)))

    #crop the image
    left = 16
    right = 240
    top = 16
    bottom = 240
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
        
    # scale to between 0 and 1
    np_image = np_image / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean) / std_dev
    
    #transpose 
    np_image = np_image.transpose((2, 0, 1))
    
    #return the converted image, ready for PyTorch
    return np_image
    
process_image(input_image)




def imshow(image, ax=None, title=None):
    #if ax is None:
    #    fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    # ax.imshow(image)
    
    return ax

# test it out
# imshow(process_image("flowers/test/1/image_06743.jpg"))


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # predict the class from an image file
    proc_image = process_image(image_path)
    
    img = torch.from_numpy(proc_image).type(torch.FloatTensor)
    
    if gpu:
        model.to('cuda')
        img = img.cuda()
    else:
        model.to('cpu')
        img = img.to('cpu')
        
    img = img.unsqueeze(0)
    
    probs = torch.exp(model.forward(img))
    top_probs, top_classes = probs.topk(topk)
    
    top_probs = top_probs.to('cpu')
    top_classes = top_classes.to('cpu')
    
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_classes = top_classes.detach().numpy().tolist()[0]
    
    #index to classes
    name_to_cat = {val: key for key, val in
                   cat_to_name.items()}
    
    top_labels = []
    
    
    for item in range(topk):
        top_classes[item] += 1
    top_labels = [cat_to_name[str(item)] for item in top_classes]
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # print("top_classes are:", top_classes)
    # print("top_labels are:", top_labels)
    

    return top_probs, top_labels, top_classes




# Display an image along with the top_K classes

#open the image
image = input_image
im = Image.open(image)

flower_num = image.split('/')[2]
np_image = np.array(im)

print()
print("Prediction results")
print()
print("Flower name:", cat_to_name[flower_num])

#display the results
probabilities, labels, classes = predict(image, model, top_k)
print()
print("Top K results")
print()

for item in range(top_k):
    print("Item:", item, "\tProbability: %.3f%%" % (probabilities[item] * 100), "\tClass: ", classes[item], "\tFlower:", labels[item])

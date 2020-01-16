#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Geoffrey Raposo
# DATE CREATED: 15.1.20                                
# REVISED DATE: 
#
##
# Imports python modules
import argparse, os
import torch
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

# Main program function
def main():
    
    args = parse_args()
    
    # Set the data directories
    checkpoint_filepath = os.getcwd() + "/" + args.checkpoint
    image_path = os.getcwd() + "/" + args.path
    
    # Set device
    device = torch.device("cpu")
    if(args.gpu and torch.cuda.is_available()):
        device = torch.device("cuda")
    
    # Restore model and optimizer
    model, optimizer = load_checkpoint(checkpoint_filepath, args.gpu)
    
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        
        img = process_image(image_path)
        img = torch.from_numpy(np.array([img])).to(device).float()
        
        logps = model(img)

        ps = torch.exp(logps)
        
        if args.top_k:
            top_k = int(args.top_k)
        else:
            top_k = 1
            
        top_p, top_class = ps.topk(top_k, dim=1)
        
        # Convert tensor to list
        top_class = top_class.tolist()[0]
        top_p = top_p.tolist()[0]
        
        # Set class names if available
        if args.category_names:
            cat_filepath = os.getcwd() + "/" + args.category_names
            with open(cat_filepath, 'r') as f:
                cat_to_name = json.load(f)
                
            # Reverse index / values
            idx_to_class = {i: c for c, i in model.class_to_idx.items()}

            for idx, c in enumerate(top_class):
                top_class[idx] = str(cat_to_name[idx_to_class[c]]).capitalize()
        
        # Print results
        print("Most probable classes")
        i = 1
        for c, p in zip(top_class, top_p):
            print('{:2}) {:>3}: {:.1f}%'.format(i, c, p*100))
            i += 1

def load_checkpoint(filepath, gpu):
    
    if gpu:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location="cpu")
    
    # Load the pretrained model
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(checkpoint['learning_rate']))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
    
    # Keep aspect ratio and set shortest side to 256
    width, height = 256, 256
    if img.size[0] > img.size [1]:
        width = img.size[0]/img.size[1]*height
    else:
        height = img.size[1]/img.size[0]*width
    
    img.thumbnail((width, height))
    
    # Center crop
    left = (img.width-224)/2
    right = left + 224
    top = (img.height-224)/2
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Color channels from 0-255 to 0-1 (expected by the model)
    np_img = np.array(img)/255
    
    # Normalize the image
    means, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    np_img = (np_img - means) / std
    
    # Reorder dimension for PyTorch
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img

def parse_args():
    # Get input arguments
    parser = argparse.ArgumentParser(description='A utility to predict the type of flower on an image.')
    parser.add_argument('path', help='The path to the image')
    parser.add_argument('checkpoint', help='The name of the checkpoint')
    parser.add_argument('--top_k', help='Return top K most likely classes')
    parser.add_argument('--category_names', help='The file containing the mapping of the categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use the gpu')
    return parser.parse_args()
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
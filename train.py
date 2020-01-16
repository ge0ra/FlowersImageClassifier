#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Geoffrey Raposo
# DATE CREATED: 3.1.20                                
# REVISED DATE: 
#
##
# Imports python modules
import argparse, os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Main program function
def main():
    
    args = parse_args()
    
    # Set the data directories
    train_dir = os.getcwd() + "/" + args.data_directory + "/train"
    valid_dir = os.getcwd() + "/" + args.data_directory + "/valid"
    
    # Define transforms for the training and validation sets
    transform, dataset, dataloader = dict(), dict(), dict()

    transform['training'] = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    transform['validation']= transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    dataset['training'] = datasets.ImageFolder(train_dir, transform=transform['training'])
    dataset['validation'] = datasets.ImageFolder(valid_dir, transform=transform['validation'])

    # Define the dataloaders
    dataloader['training'] = torch.utils.data.DataLoader(dataset['training'], batch_size=64, shuffle=True)
    dataloader['validation'] = torch.utils.data.DataLoader(dataset['validation'], batch_size=64, shuffle=True)

    # Load the pretrained model
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Define number of inputs
    if(args.arch == 'densenet121'):
        nb_input = model.classifier.in_features
    elif(args.arch == 'vgg13'):
        nb_input = model.classifier[0].in_features
    else:
        nb_input = 0
        
    # Define classifier architecture    
    model.classifier = nn.Sequential(nn.Linear(nb_input, int(args.hidden_units)),
                                     nn.ReLU(),
                                     nn.Dropout(0.4),
                                     nn.Linear(int(args.hidden_units), 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))

    # Train the model
    train_model(model, optimizer, criterion, dataloader, int(args.epochs), float(args.learning_rate), args.gpu)
    
    # Save a checkpoint
    save_checkpoint(args.save_dir, model, optimizer, criterion, args.arch, float(args.learning_rate), dataset['training'].class_to_idx)
    
def train_model(model, optimizer, criterion, dataloader, epochs, learning_rate, gpu):
    
    # Use GPU if available and requested
    device = torch.device("cpu")
    if(torch.cuda.is_available() and gpu):
        device = torch.device("cuda")
        
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 10

    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in dataloader['training']:
            steps += 1

            # Move the inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader['validation']:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        loss = criterion(logps, labels)

                        test_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)

                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                #train_losses.append(running_loss/print_every)
                #valid_losses.append(test_loss/len(dataloader['validation']))

                print(f"Epoch: {epoch+1}/{epochs}",
                      f"Train loss: {running_loss/print_every:.3f}.. ",
                      f"Test Loss: {test_loss/len(dataloader['validation']):.3f}.. ",
                      f"Test Accuracy: {accuracy/len(dataloader['validation'])*100:.3f}% ")

                running_loss = 0
                model.train()
                
def save_checkpoint(folder, model, optimizer, criterion, arch, learning_rate, class_to_idx):
    checkpoint = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'classifier': model.classifier,
             'criterion': criterion,
             'arch': arch,
             'learning_rate': learning_rate,
             'class_to_idx': class_to_idx}

    torch.save(checkpoint, folder + '/checkpoint.pth')
    print('Checkpoint saved.')
    
def parse_args():
    # Get input arguments
    parser = argparse.ArgumentParser(description='A utility to train a new network on a dataset')
    parser.add_argument('data_directory', help='The directory containing the data')
    parser.add_argument('--save_dir', default='checkpoints', help='The directory to save checkpoints')
    parser.add_argument('--arch', default='densenet121', help='The architecture of the model')
    parser.add_argument('--learning_rate', default=0.003, help='The learning rate')
    parser.add_argument('--hidden_units', default=500, help='The number of hidden units')
    parser.add_argument('--epochs', default=3, help='The number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use the gpu')
    return parser.parse_args()
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
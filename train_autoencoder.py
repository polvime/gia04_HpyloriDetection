import pandas as pd
import cv2
import os
import time, copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np

cropped_csv = pd.read_csv("/export/fhome/mapsiv/QuironHelico/CroppedPatches/metadata.csv")
annotated_csv = pd.read_csv("annotated_patients.csv")
dont_use = list(annotated_csv['PatientID'])

#load images and store them along with its corresponding file name

cropped_patients = {}

#go through the Annotated_Patches folder and open each folder to look for the images
for filename in os.listdir('/export/fhome/mapsiv/QuironHelico/CroppedPatches'):
    
    #if the file is a folder
    if os.path.isdir('/export/fhome/mapsiv/QuironHelico/CroppedPatches/' + filename) and filename[:-2] not in dont_use:
        cropped_patients[filename] = [[], 0]
        #go through the folder and open each image
        for image in os.listdir('/export/fhome/mapsiv/QuironHelico/CroppedPatches/' + filename):
            #read the image
            #img = cv2.imread('/import/fhome/mapsiv/QuironHelico/CroppedPatches/' + filename + '/' + image)
            image_name = filename + '.' + image[:-4] #take .png off the image name
            #store the image in the dictionary
            cropped_patients[filename][0].append(str(image))

auto_imgs = []
for key, values in cropped_patients.items():
    label = list(cropped_csv.loc[cropped_csv["CODI"] == key[:-2]]["DENSITAT"])[0]
    if label == 'NEGATIVA':
        values[1] = -1
    elif label == 'BAIXA':
        values[1]  = 1
    elif label == 'ALTA':
        values[1]  = 2
    else:
        values[1]  = 0

all_healthy_patients = [key for key, value in cropped_patients.items() if value[1] == -1] 
low_patients = [key for key, value in cropped_patients.items() if value[1] == 1] 
high_patients = [key for key, value in cropped_patients.items() if value[1] == 2] 
auto_pat = all_healthy_patients[min(len(all_healthy_patients), len(low_patients), len(high_patients)):]
healthy_patients = all_healthy_patients[:min(len(all_healthy_patients), len(low_patients), len(high_patients))]

auto_imgs = []

for key, values in cropped_patients.items():
    if key in auto_pat:
        for img in values[0]:
            path = key + '/' + img
            auto_imgs.append(path)

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        #get the image and its corresponding label
        image = cv2.imread('/export/fhome/mapsiv/QuironHelico/CroppedPatches/' + self.data[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #resize image to 256 x 256
        image = cv2.resize(image, (256, 256))

        #convert the image to a tensor
        image = torch.from_numpy(image).float()

        #reshape the image to (1, 3, 256, 256)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        return image

dataset = Dataset(auto_imgs)
batch_size = 64
#split the dataset into training and validation and test sets

if (int(len(dataset)*0.8)+int(len(dataset)*0.2)) == len(dataset):
    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
else:
    diference = len(dataset) - (int(len(dataset)*0.8)+int(len(dataset)*0.2))
    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8)+diference, int(len(dataset)*0.2)])

#load the training and testing sets into dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

#check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        #encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        #decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU()
        )

    def forward(self, x):
                            # [batch_size, 3, 256, 256]
        #encoder
        x = self.enc1(x)    #[batch_size, 32, 256, 256]
        x = self.enc2(x)    #[batch_size, 64, 128, 128]
        x = self.enc3(x)    #[batch_size, 64, 64, 64]

        #decoder
        x = self.dec1(x)    #[batch_size, 64, 128, 128]
        x = self.dec2(x)    #[batch_size, 32, 256, 256]
        x = self.dec3(x)    #[batch_size, 3, 256, 256]

        return x
    
#Function to train our model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()    
    losses = {"train": [], "val": []}   
    final_losses = {"train": [], "val": []} 

    # we will keep a copy of the best weights so far according to validation loss
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0 
            # Iterate over data.
            for window in dataloaders[phase]:
                img = window[0]
                img = img.to(device)               

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss                    
                    output = model(img)    
                    
                    loss = criterion(output, img)
                    losses[phase].append(loss.item())                   

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                running_loss += loss.item()        

            epoch_loss = running_loss / len(dataloaders[phase])
            final_losses[phase].append(epoch_loss)
            #print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, final_losses 

model = Autoencoder()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-5)

num_epochs = 15

dataloaders_dict = {}
dataloaders_dict['train'] = train_loader
dataloaders_dict['val'] = val_loader

#Train the model
model, losses = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

torch.save(model.state_dict(), 'auto.pth')

# Extract the loss values and plot them
#We begin at the tenth epoch because the first losses were too big and the scale of the plot was not useful
train_l = losses['train'][:]
val_l = losses['val'][:]
epochs = range(0, len(losses['train'][:]))

# Plot epoch vs training loss until epoch 100
plt.plot(epochs, train_l, label='Training Loss')
plt.plot(epochs, val_l, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over training')
plt.legend()
plt.show()
plt.savefig('loss.png')
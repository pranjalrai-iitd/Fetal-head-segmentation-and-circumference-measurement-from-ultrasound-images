import torch
import glob
import numpy as np
from PIL import Image


dir = 'C:/Users/pranjal/Desktop/Project/Deep Learning/Fetal head circumference/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Loads to the GPU 
x_size = (572, 572)
y_size = (572, 572)

class Data(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):

        self.x_train = sorted(glob.glob(dir+'Dataset/Segmentation/train/*HC.png'))
        self.x_val = sorted(glob.glob(dir+'Dataset/Segmentation/validate/*HC.png'))
        self.x_test = sorted(glob.glob(dir+'test_set/*HC.png'))
        self.y_train = sorted(glob.glob(dir+'Dataset/Segmentation/train/*_Annotation.png'))
        self.y_val = sorted(glob.glob(dir+'Dataset/Segmentation/validate/*_Annotation.png'))
        self.y_test = sorted(glob.glob(dir+'test_set/*HC_Annotation.png'))
        self.data = data


    def __len__(self):

        if(self.data == 'train'):
            return len(self.x_train)

        elif(self.data == 'validate'):
            return len(self.x_val)

        elif(self.data == 'test'):
            return len(self.x_test)

        else:
            return ValueError("No data")



    def __getitem__(self, idx):

        if(self.data == 'train'):
            x = np.array(Image.open(self.x_train[idx]).convert("L").resize(x_size)).reshape(1, 572, 572)
            y = np.array(Image.open(self.y_train[idx]).convert("L").resize(y_size)).reshape(1, 572, 572) # Convert to gray scale
            return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)

        elif(self.data == 'validate'):
            x = np.array(Image.open(self.x_val[idx]).convert("L").resize(x_size)).reshape(1, 572, 572)
            y = np.array(Image.open(self.y_val[idx]).convert("L").resize(y_size)).reshape(1, 572, 572) # Convert to gray scale
            return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)

        elif(self.data == 'test'):
            x = np.array(Image.open(self.x_test[idx]).convert("L").resize(x_size)).reshape(1, 572, 572)
            y = np.array(Image.open(self.y_test[idx]).convert("L").resize(y_size)).reshape(1, 572, 572) # Convert to gray scale
            return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)

        else:
            return ValueError("No data")

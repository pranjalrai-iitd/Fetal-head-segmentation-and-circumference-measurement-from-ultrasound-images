import numpy as np
import torch
import torch.optim as optim
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from Data import Data
from Unet import Unet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dice_sim(pred, truth):
    epsilon = 1e-8
    num_batches = pred.size(0)
    m1 = pred.view(num_batches, -1).bool()
    m2 = truth.view(num_batches, -1).bool()

    intersection = torch.logical_and(m1, m2).sum(dim=1)
    return (((2. * intersection + epsilon) / (m1.sum(dim=1) + m2.sum(dim=1) + epsilon)).sum(dim=0))/2

train = Data('train', None)
val = Data('validate', None)
test = Data('test', None)

num_epochs = 1
batch_size = 2

train_generator = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, num_workers=0)
val_generator = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True, num_workers=0)
test_generator = torch.utils.data.DataLoader(test, batch_size=2, shuffle=True, num_workers=0)

model = Unet().to(device)
#model = torch.load(dir+"model", map_location=torch.device("cpu")) To load pretrained model

loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)
train_size = len(train)
val_size = len(val)

for epoch in range(num_epochs):

    running_loss = 0.0
    val_loss = 0.0
    val_acc = 0.0

    # Training
    for i, data in enumerate(train_generator):
        image, truth = data
        truth = truth/255
        predictions = model(image)
        loss = loss_fn(predictions, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # Validation
    with torch.no_grad():
        for i, data in enumerate(val_generator):
            image, truth = data
            truth = truth/255
            predictions = model(image)
            predictions = torch.round(predictions) # Rounding the predictions for classification
            loss = loss_fn(predictions, truth)
            val_loss += loss.item()
            val_acc += dice_sim(predictions, truth)*100

    epoch_train_loss = running_loss / (train_size//batch_size+1)
    epoch_val_loss = val_loss / (val_size//batch_size+1)
    epoch_val_acc = val_acc / (val_size//batch_size+1)
    scheduler.step(epoch_val_loss) # LR Scheduler
    print(f"==>train_loss: {epoch_train_loss} ==>val_loss: {epoch_val_loss} ==>val_accuracy: {epoch_val_acc}")

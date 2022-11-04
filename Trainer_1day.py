import datasetHandler_1day
import LSTMdataset
import VS_LSTM
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import exists
import math
from scipy import stats
import os

def saveModel():
    with open("models/1day_"+stock+".pth", "w") as c:
        pass
    checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "batch_number" : batchNum,
    }
    FILE = "models/1day_"+stock+".pth"
    torch.save(checkpoint, FILE)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
def strToList(st): #WARNING: THIS FUNCTION IS DIFFERENT FROM THE OTHER strToList functions...
    if st == '[]':
        return []
    factor = -1
    for ch in st:
        if ch != '[':
            break
        factor += 1
    if factor == 0:
        return [int(x) if isfloat(x) else x for x in st.split("[")[1].split("]")[0].split(", ")]

    sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
    lst = []
    for s in sList:
        lst.append(strToList(s))
    return lst

def signal_handler(sig, frame):
    with open("models/1day_"+stock+".pth", "w") as c:
        pass
    checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "batch_number" : batchNum,
    }
    FILE = "models/1day_"+stock+".pth"
    torch.save(checkpoint, FILE)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

stock = input("Enter stock name: ")
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

with open("settings/1day_"+stock+".txt", 'r') as s:
    for line in s:
        line = line.split("\n")[0]
        data = line.split(" = ")[1]
        tag = line.split(" = ")[0]
        if tag == "learning_rate":
            learning_rate = float(data)
        elif tag == "num_epochs":
            num_epochs = int(data)
        elif tag == "num_layers":
            num_layers = int(data)
        elif tag == "batchSize":
            batchSize = int(data)
        elif tag == "printingBatch":
            printingBatch = int(data)

#Settings:
criterion = nn.MSELoss().to(device)

dataset = LSTMdataset.LSTMdataset("datasets", "1day_"+stock+"_MD.txt")
#total_samples = len(dataset)
#n_iterations = math.ceil(total_samples/batchSize)
inputSize = len(dataset[0][0][0])
model = VS_LSTM.LSTM(num_layers, 13+(2*inputSize), inputSize)
#model = LSTM.LSTM(1, len(dataset[0][0]), inputSize)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

ch1 = input("Start training from scratch? Y/N: ")

if torch.cuda.is_available():
    if ch1.upper() == "Y":
        with open("logs/1day_"+stock+"_log.out", "w") as lg:
            lg.write("Device name: ")
            lg.write(str(torch.cuda.get_device_name(0))+"\n")
    else:
        with open("logs/1day_"+stock+"_log.out", "a") as lg:
            lg.write("\nDevice name: ")
            lg.write(str(torch.cuda.get_device_name(0))+"\n")
else:
    if ch1.upper() == "Y":
        with open("logs/1day_"+stock+"_log.out", "w") as lg:
            pass
        
epoch = 0
batchNum = 0
loss = "Dummy Initialisation"

ch = input("Use existing datasets? Y/N: ")

if ch.upper() == "N":
    ds = datasetHandler_1day.datasetHandler()
    windowSize = int(input("Enter the windowSize: "))
    ds.createDataset(stock, windowSize)
    ds.shuffleDataset("datasets", "1day_"+stock+"_MD.txt")
    
    dataset = LSTMdataset.LSTMdataset("datasets", "1day_"+stock+"_MD.txt")
    #total_samples = len(dataset)
    #n_iterations = math.ceil(total_samples/batchSize)
    inputSize = len(dataset[0][0][0])
    model = VS_LSTM.LSTM(num_layers, 13+(2*inputSize), inputSize)
    #model = LSTM.LSTM(1, len(dataset[0][0]), inputSize)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    
    with open("models/1day_"+stock+".pth", "w") as c:
        pass
else:
    if ch1.upper() == "Y":
        with open("models/1day_"+stock+".pth", "w") as c:
            pass
        with open("logs/1day_"+stock+"_TrainingLoss.txt", "w") as tl:
            pass
    else:
        FILE = "models/1day_"+stock+".pth"
        checkpoint = torch.load(FILE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        epoch = checkpoint['epoch']
        batchNum = checkpoint['batch_number']
        with open("models/1day_"+stock+".pth", "w") as c:
            pass

with open("logs/1day_"+stock+"_log.out", "a") as lg:
    lg.write("\nStarting from:\n")
    lg.write("epoch = "+str(epoch)+"\n")
    lg.write("batchNum = "+str(batchNum)+"\n")
    lg.write("batchSize = "+str(batchSize)+"\n\n")

train_loader = DataLoader(dataset=dataset,
                      batch_size=batchSize,
                      shuffle=False,
                      num_workers=0)      

while(epoch < num_epochs):    

    ##########################################################
    for i, (inputs, labels, seqLens) in enumerate(train_loader):
        if i == batchNum:
            # Forward pass and loss
            y_pred = model(inputs, seqLens)
            
            #y_pred = model(inputs)
            #y_pred = y_pred.view(y_pred.size(0))
            
            #labels = labels.view(labels.size(0))
            #labels = labels.long()
            
            loss = criterion(y_pred, labels)
            if batchNum == printingBatch:
                with open("logs/1day_"+stock+"_TrainingLoss.txt", "a") as tl:
                    tl.write("Epoch : "+str(epoch)+"  BatchNum : "+str(i)+"  Loss : "+str(loss.item())+"\n")
                with open("logs/1day_"+stock+"_log.out", "a") as lg:
                    lg.write("Epoch : "+str(epoch)+"  BatchNum : "+str(i)+"  Loss : "+str(loss.item())+"\n")
                    lg.write("\n")
                    lg.write("y_pred:\n")
                    lg.write(str(y_pred)+"\n")
                    lg.write("\n")
                    lg.write("labels:\n")
                    lg.write(str(labels)+"\n")
                    lg.write("\n\n")
            
            
            # Backward pass and update
            loss.backward()
            optimizer.step()  
                          
            # zero grad before new step
            optimizer.zero_grad()
            
            batchNum += 1

    batchNum = 0
    epoch += 1

signal_handler(0, 0)
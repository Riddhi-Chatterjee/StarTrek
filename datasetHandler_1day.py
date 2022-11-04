import time
import signal
import sys
from os.path import exists
import os
import torch
#import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import shutil
#import openpyxl
from pathlib import Path

class datasetHandler(object):
    
    def readSourceFile(self, filename):
        numIntervalsInDay = 75
        with open(filename, "r") as csv:
            data = []
            csv.readline()
            for line in csv:
                line = line.split("\n")[0]
                tmp = [float(x) for x in line.split(",")[1:]]
                data.append(tmp)
        return data

    def splitData(self, data, window):
        n = len(data)
        train = []
        test = []
        for i in range(n-window+1):
            train.append(data[i:i+window-1])
            test.append(data[i+window-1])
        return (train, test)

    def writeDestFile(self, filename, train, test):
        n = len(train)
        with open(filename, "w") as txt:
            for i in range(n):
                line = str(test[i]) + ":" + str(train[i]) + "\n"
                txt.write(line)

    def createDataset(self, stockname, windowsize):
        sourcefile = self.sourceFile(stockname)
        destinationfile = self.destinationFile(stockname)
        rawdata = self.readSourceFile(filename=sourcefile)
        (train, test) = self.splitData(data=rawdata, window=windowsize)
        self.writeDestFile(destinationfile, train, test)

    def sourceFile(self, stockname): 
        filename = "datasets/1day_" + stockname + ".csv"
        return filename

    def destinationFile(stockname):
        filename = "datasets/1day_" + stockname + "_MD.txt"
        return filename

    def shuffleDataset(self, folder, filename): #Shuffle the lines in datasets/master_dataset.txt
                                #We can't store all the contents of datasets/master_dataset.txt in some list due to memory constraint
            
            lines = open(folder+"/"+filename).readlines()
            random.shuffle(lines)
            open(folder+"/"+filename,'w').writelines(lines)

class LSTMdataset(Dataset):
    
    def __init__(self, folder, filename):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        xy = np.loadtxt(folder+"/"+filename, delimiter=":", dtype = str)
        if len(xy.shape) == 1:
            xy = xy.reshape((1,xy.shape[0]))
        self.n_samples = xy.shape[0]
        seqList = []
        for seq in xy[:, 1:]:
            seqList.append(self.strToList(seq[0]))

        # here the first column is the class label, the rest is the frame sequence
        #self.x_data = torch.tensor(seqList, dtype=torch.float32) # size [n_samples, n_time_steps, n_features]
        self.l_data = [len(seq) for seq in seqList]
        self.x_data = self.padData(seqList)
        self.y_data = torch.tensor([self.strToList(x) for x in xy[:, 0]]).to(self.device) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.l_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def strToList(self, st):
        if st == '[]':
            return []
        factor = -1
        for ch in st:
            if ch != '[':
                break
            factor += 1
        if factor == 0:
            return [float(x) for x in st.split("[")[1].split("]")[0].split(", ")]
        
        sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
        lst = []
        for s in sList:
            lst.append(self.strToList(s))
        return lst
    
    def padData(self, X_list):
        max_len = 0
        num_features = 0
        for seq in X_list:
            if len(seq) != 0:
                num_features = len(seq[0])
            if len(seq) > max_len:
                max_len = len(seq)

        padList = [0]*num_features

        for i in range(len(X_list)):
            iter = max_len - len(X_list[i])
            for j in range(iter):
                X_list[i].append(padList)

        X = torch.tensor(X_list, dtype = torch.float32).to(self.device)

        #print(X)
        return X
    
if __name__ == "__main__":
    stockname = str(input())
    windowsize = int(input())
    dh = datasetHandler()
    dh.createDataset(stockname, windowsize)
    dataset = LSTMdataset("datasets", "5mins_APPLE_MD.txt")
    print(len(dataset))

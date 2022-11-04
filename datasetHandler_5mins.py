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
            visitedLines = 0
            data = []
            csv.readline()
            for line in csv:
                line = line.split("\n")[0]
                tmp = [float(x) for x in line.split(", ")[1:]]
                if len(tmp) == numIntervalsInDay:
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
        filename = "datasets/5mins_" + stockname + ".csv"
        return filename

    def destinationFile(self, stockname):
        filename = "datasets/5mins_" + stockname + "_MD.txt"
        return filename
    
    def shuffleDataset(self, folder, filename): #Shuffle the lines in datasets/master_dataset.txt
                              #We can't store all the contents of datasets/master_dataset.txt in some list due to memory constraint
        
        lines = open(folder+"/"+filename).readlines()
        random.shuffle(lines)
        open(folder+"/"+filename,'w').writelines(lines)

    
if __name__ == "__main__":
    stockname = str(input("Enter stock name: "))
    windowsize = int(input("Enter the window size: "))
    dh = datasetHandler()
    dh.createDataset(stockname, windowsize)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import signal
import sys

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class LSTM(nn.Module):
    def __init__(self, nb_layers, hidden_size, inputSize):
        super(LSTM, self).__init__()   
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.nb_layers = nb_layers
        self.hidden_size = hidden_size
        self.input_size = inputSize

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # design LSTM
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.nb_layers,
            batch_first=True,
        ).to(self.device)

        # linear layer:
        self.fully_connected = nn.Linear(self.hidden_size, self.input_size).to(self.device)
        
        #Sigmoid layer:
        self.sigmoid = nn.Sigmoid().to(self.device)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, hidden_size)
        hidden_a = torch.rand(self.nb_layers, self.batch_size, self.hidden_size).to(self.device)
        hidden_b = torch.rand(self.nb_layers, self.batch_size, self.hidden_size).to(self.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def getInputs(self, X, X_lengths):
        '''
        max_len = 0
        X_lengths = []
        num_features = 0
        for seq in X_list:
            X_lengths.append(len(seq))
            if len(seq) != 0:
                num_features = len(seq[0])
            if len(seq) > max_len:
                max_len = len(seq)
                
        padList = [0]*num_features
        
        for i in range(len(X_list)):
            iter = max_len - len(X_list[i])
            for j in range(iter):
                X_list[i].append(padList)
        
        X = torch.tensor(X_list, dtype = torch.float32)
        '''
        
        X = X.to(self.device)
        t = torch.tensor(X_lengths).to(self.device)
        sorted, indices = torch.sort(t, descending=True)
        X_lengths = [int(x) for x in sorted]
        X_copy = torch.clone(X)
        for ind1, ind2 in enumerate(indices):
            X[int(ind1)] = X_copy[int(ind2)]

        #print(X)
        #print(X_lengths)
        return [X, X_lengths, indices]

    def forward(self, X_padded, X_lengths):
        X, X_lengths, indices = self.getInputs(X_padded, X_lengths)
        self.batch_size, seq_len, _ = X.size()
        
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        
        self.hidden = self.init_hidden()

        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_size)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        out = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        out, self.hidden = self.lstm(out, self.hidden)

        # undo the packing operation
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.fully_connected(out[:, out.shape[1]-1,:])
        #out = out.view(out.shape[0])
        
        #out = (self.sigmoid(out) * 40) + 10
        #out = (torch.tanh(out) * 20) + 30
        
        out_copy = torch.clone(out)
        for ind1, ind2 in enumerate(indices):
            out[int(ind2)] = out_copy[int(ind1)]

        return out
    
def main():
    inputSize = 75
    model_5mins = LSTM(4, 2*inputSize, inputSize)
    
    inputSize = 1
    model_5mins_Gran = LSTM(4, 13+(2*inputSize), inputSize)
    
    inputSize = 6
    model_1day = LSTM(4, 13+(2*inputSize), inputSize)
    
    X = torch.tensor([[[1, 20, 45, -1, 65, 87], [-1, 25, 50, 1, 78, 89], [1, 30, 55, -1, 86, 78], [-1, 35, 60, 1, 8, 66]], [[1, 20, 45, -1, 57, 75], [1, 15, 40, -1, 8, 78], [1, 10, 35, -1, 89, 8], [0, 0, 0, 0, 0, 0]]], dtype=torch.float32)
    X_lengths = [4, 4]
    
    print("")
    print(model_1day(X, X_lengths))
    print("")
    
if __name__ == "__main__":
    main()
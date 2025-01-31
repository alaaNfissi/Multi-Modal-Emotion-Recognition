from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_model.xlstm import mLSTM, sLSTM


class CNN18XLSTM(nn.Module):
    """
          A convolutional neural network (CNN) with xLSTM for emotion recognition from raw audio.

        Args:
            n_input (int): Number of input channels (e.g., 1 for mono audio).
            hidden_dim (int): Dimension of the hidden state in LSTM layers.
            n_layers (int): Number of LSTM layers.
            n_output (int, optional): Number of output classes for classification. Defaults to None.
            stride (int, optional): Stride for the first convolution layer. Defaults to 4.
            n_channel (int, optional): Number of channels in the convolutional layers. Defaults to 18.
      """
    def __init__(self, n_input, hidden_dim, n_layers, n_output=None, stride=4, n_channel=18):
        super().__init__()


        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel) #pour accelerer l'apprentissage et le stabiliser
        self.relu1 = nn.LeakyReLU()

        self.pool1 = nn.MaxPool1d(4, stride=None)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm1d(n_channel)
        self.relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm1d(n_channel)
        self.relu5 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.05)

        self.pool2 = nn.MaxPool1d(4, stride=None)

        self.conv6 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm1d(2 * n_channel)
        self.relu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm1d(2 * n_channel)
        self.relu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm1d(2 * n_channel)
        self.relu8 = nn.LeakyReLU()

        self.conv9 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn9 = nn.BatchNorm1d(2 * n_channel)
        self.relu9 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.05)

        self.pool3 = nn.MaxPool1d(4, stride=None)

        self.conv10 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn10 = nn.BatchNorm1d(4 * n_channel)
        self.relu10 = nn.LeakyReLU()

        self.conv11 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn11 = nn.BatchNorm1d(4 * n_channel)
        self.relu11 = nn.LeakyReLU()

        self.conv12 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn12 = nn.BatchNorm1d(4 * n_channel)
        self.relu12 = nn.LeakyReLU()

        self.conv13 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn13 = nn.BatchNorm1d(4 * n_channel)
        self.relu13 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.1)

        self.pool4 = nn.MaxPool1d(4, stride=None)

        self.conv14 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn14 = nn.BatchNorm1d(8 * n_channel)
        self.relu14 = nn.LeakyReLU()

        self.conv15 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn15 = nn.BatchNorm1d(8 * n_channel)
        self.relu15 = nn.LeakyReLU()

        self.conv16 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn16 = nn.BatchNorm1d(8 * n_channel)
        self.relu16 = nn.LeakyReLU()

        self.conv17 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn17 = nn.BatchNorm1d(8 * n_channel)
        self.relu17 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(8 * n_channel, 4 * n_channel)
        self.relu18 = nn.LeakyReLU()




        mlstm_par = {
            'inp_dim' : 4 * n_channel,
            'head_dim' : 18,
            'head_num' : 4,
            'p_factor' : 2,
            'ker_size' : 4,
        }

        slstm_par = {
            'inp_dim' : 4 * n_channel,
            'head_dim' : 18,
            'head_num' : 4,
            'p_factor' : 4/3,
            'ker_size' : 4,
        }


        which =[True,True,True]
        

        self.xlstm : List[mLSTM | sLSTM] = nn.ModuleList([
            mLSTM(**mlstm_par) if w else sLSTM(**slstm_par)
            for w in which
        ])


        
        self.fc2 = nn.Linear(4 * n_channel, n_output)
        self.relu19 = nn.LeakyReLU()

    def forward(self, x):

        """
        Passes input through several convolutional layers, followed by xLSTM layers (forward) 
        for sequence modeling, and returns the final output for classification.

        Args:
            x (torch.Tensor): Input tensor with shape `[batch_size, seq_length]`.

        Returns:
            torch.Tensor: Final output tensor for classification.
        """
        # x [b,  i (16K)]
        x = x.unsqueeze(1) #pour le canal  [b, c=1 , i (16K)]

        x = self.conv1(x) # [b, c=18 , i 3961]
        x = self.relu1(self.bn1(x)) # [b, c=18 , i 3961]

        x = self.pool1(x) # [b, c=18 , i=990]

        x = self.conv2(x)
        x = self.relu2(self.bn2(x)) # [b, c=18 , i=990]

        x = self.conv3(x)
        x = self.relu3(self.bn3(x)) # [b, c=18 , i=990]

        x = self.conv4(x)
        x = self.relu4(self.bn4(x)) # [b, c=18 , i=990]


        x = self.conv5(x)
        x = self.relu5(self.bn5(x)) # [b, c=18 , i=990]
        x= self.dropout1(x)
        x = self.pool2(x) # [b, c=18 , i=247]


        x = self.conv6(x)
        x = self.relu6(self.bn6(x))  # [b, c=36 , i=247]


        x = self.conv7(x)
        x = self.relu7(self.bn7(x)) # [b, c=36 , i=247]


        x = self.conv8(x)
        x = self.relu8(self.bn8(x))# [b, c=36 , i=247]


        x = self.conv9(x)
        x = self.relu9(self.bn9(x)) # [b, c=36 , i=247]
        x= self.dropout2(x)

        x = self.pool3(x) # [b, c=36 , i=61]


        x = self.conv10(x)
        x = self.relu10(self.bn10(x))  # [b, c=72 , i=61]


        x = self.conv11(x)
        x = self.relu11(self.bn11(x)) # [b, c=72 , i=61]


        x = self.conv12(x)
        x = self.relu12(self.bn12(x)) # [b, c=72 , i=61]


        x = self.conv13(x)
        x = self.relu13(self.bn13(x)) # [b, c=72 , i=61]
        x= self.dropout3(x)

        x = self.pool4(x) # [b, c=72 , i=15]


        x = self.conv14(x)
        x = self.relu14(self.bn14(x)) # [b, c=144 , i=15]


        x = self.conv15(x)
        x = self.relu15(self.bn15(x)) # [b, c=144 , i=15]


        x = self.conv16(x)
        x = self.relu16(self.bn16(x)) # [b, c=144 , i=15]


        x = self.conv17(x)
        x = self.relu17(self.bn17(x)) # [b, c=144 , i=15]
        x= self.dropout4(x)

        # AVG : [b, c=144 , 1]
        #x = F.avg_pool1d(x, x.shape[-1])  #reduces the remaining temporal dimension to a single value for each channel, summarizing the entire sequence.
        
        #c -> innput size et i -> seq len
        x = x.permute(2, 0, 1) # reshape  [15, b , 144]

        x = self.fc1(self.relu18(x)) # reduces redundancy and preparing the data for classification.

        # [15, b , 72]
        
        hid = [l.init_hidden(x.size(1)) if isinstance(l, mLSTM) else l.init_hidden(x.size(1)) for l in self.xlstm]

        # Pass the sequence through the mLSTM and sLSTM blocks
        out = []

        # Compute model output and update the hidden states

        for seq in x:
            for i, lstm in enumerate(self.xlstm):

                seq, hid[i] = lstm(seq, hid[i])

                out.append(seq)
        batch_first=True
        out = torch.stack(out, dim=1 if batch_first else 0)

        
        # x = self.fc2(self.relu19(out[:,-1])) #projects the output from the GRU to the number of output classes (n_output).

        return out[:,-1]
    

    class CNN18biXLSTM(nn.Module):
      """
          A convolutional neural network (CNN) with a bi-directional xLSTM for emotion recognition from raw audio.

        Args:
            n_input (int): Number of input channels (e.g., 1 for mono audio).
            hidden_dim (int): Dimension of the hidden state in LSTM layers.
            n_layers (int): Number of LSTM layers.
            n_output (int, optional): Number of output classes for classification. Defaults to None.
            stride (int, optional): Stride for the first convolution layer. Defaults to 4.
            n_channel (int, optional): Number of channels in the convolutional layers. Defaults to 18.
      """
      def __init__(self, n_input, hidden_dim, n_layers, n_output=None, stride=4, n_channel=18):
        super().__init__()


        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel) #pour accelerer l'apprentissage et le stabiliser
        self.relu1 = nn.LeakyReLU()

        self.pool1 = nn.MaxPool1d(4, stride=None)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm1d(n_channel)
        self.relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm1d(n_channel)
        self.relu5 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.15)

        self.pool2 = nn.MaxPool1d(4, stride=None)

        self.conv6 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm1d(2 * n_channel)
        self.relu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm1d(2 * n_channel)
        self.relu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm1d(2 * n_channel)
        self.relu8 = nn.LeakyReLU()

        self.conv9 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn9 = nn.BatchNorm1d(2 * n_channel)
        self.relu9 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.15)

        self.pool3 = nn.MaxPool1d(4, stride=None)

        self.conv10 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn10 = nn.BatchNorm1d(4 * n_channel)
        self.relu10 = nn.LeakyReLU()

        self.conv11 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn11 = nn.BatchNorm1d(4 * n_channel)
        self.relu11 = nn.LeakyReLU()

        self.conv12 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn12 = nn.BatchNorm1d(4 * n_channel)
        self.relu12 = nn.LeakyReLU()

        self.conv13 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn13 = nn.BatchNorm1d(4 * n_channel)
        self.relu13 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.15)

        self.pool4 = nn.MaxPool1d(4, stride=None)

        self.conv14 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn14 = nn.BatchNorm1d(8 * n_channel)
        self.relu14 = nn.LeakyReLU()

        self.conv15 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn15 = nn.BatchNorm1d(8 * n_channel)
        self.relu15 = nn.LeakyReLU()

        self.conv16 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn16 = nn.BatchNorm1d(8 * n_channel)
        self.relu16 = nn.LeakyReLU()

        self.conv17 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn17 = nn.BatchNorm1d(8 * n_channel)
        self.relu17 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(0.15)
        self.pool5 = nn.MaxPool1d(4, stride=None)
       
        self.fc1 = nn.Linear(8 * n_channel, 4 * n_channel)
        self.relu18 = nn.LeakyReLU()
        



        mlstm_par = {
            'inp_dim' : 4 * n_channel,
            'head_dim' : 18,
            'head_num' : 4,
            'p_factor' : 2,
            'ker_size' : 4,
        }

        slstm_par = {
            'inp_dim' : 4 * n_channel,
            'head_dim' : 18,
            'head_num' : 4,
            'p_factor' : 4/3,
            'ker_size' : 4,
        }


        
        which =[True,True,True] # 3 mLSTM

        self.xlstm_f : List[mLSTM | sLSTM] = nn.ModuleList([
            mLSTM(**mlstm_par) if w else sLSTM(**slstm_par)
            for w in which
        ])


       

        
        self.fc2 = nn.Linear(4 * n_channel, n_output)
        self.relu19 = nn.LeakyReLU()

    def forward(self, x):
        """
        Passes input through several convolutional layers, followed by bi-xLSTM layers (forward and backward) 
        for sequence modeling, and returns the final output for classification.

        Args:
            x (torch.Tensor): Input tensor with shape `[batch_size, seq_length]`.

        Returns:
            torch.Tensor: Final output tensor for classification.
        """


        # x [b,  i (16K)]
        x = x.unsqueeze(1) #pour le canal  [b, c=1 , i (16K)]

        x = self.conv1(x) # [b, c=18 , i 3961]
        x = self.relu1(self.bn1(x)) # [b, c=18 , i 3961]

        x = self.pool1(x) # [b, c=18 , i=990]

        x = self.conv2(x)
        x = self.relu2(self.bn2(x)) # [b, c=18 , i=990]

        x = self.conv3(x)
        x = self.relu3(self.bn3(x)) # [b, c=18 , i=990]

        x = self.conv4(x)
        x = self.relu4(self.bn4(x)) # [b, c=18 , i=990]


        x = self.conv5(x)
        x = self.relu5(self.bn5(x)) # [b, c=18 , i=990]
        x= self.dropout1(x)

        x = self.pool2(x) # [b, c=18 , i=247]


        x = self.conv6(x)
        x = self.relu6(self.bn6(x))  # [b, c=36 , i=247]


        x = self.conv7(x)
        x = self.relu7(self.bn7(x)) # [b, c=36 , i=247]


        x = self.conv8(x)
        x = self.relu8(self.bn8(x))# [b, c=36 , i=247]


        x = self.conv9(x)
        x = self.relu9(self.bn9(x)) # [b, c=36 , i=247]
        x= self.dropout2(x)


        x = self.pool3(x) # [b, c=36 , i=61]


        x = self.conv10(x)
        x = self.relu10(self.bn10(x))  # [b, c=72 , i=61]


        x = self.conv11(x)
        x = self.relu11(self.bn11(x)) # [b, c=72 , i=61]


        x = self.conv12(x)
        x = self.relu12(self.bn12(x)) # [b, c=72 , i=61]


        x = self.conv13(x)
        x = self.relu13(self.bn13(x)) # [b, c=72 , i=61]
        x= self.dropout3(x)


        x = self.pool4(x) # [b, c=72 , i=15]


        x = self.conv14(x)
        x = self.relu14(self.bn14(x)) # [b, c=144 , i=15]


        x = self.conv15(x)
        x = self.relu15(self.bn15(x)) # [b, c=144 , i=15]


        x = self.conv16(x)
        x = self.relu16(self.bn16(x)) # [b, c=144 , i=15]


        x = self.conv17(x)
        x = self.relu17(self.bn17(x)) # [b, c=144 , i=15]
        x= self.dropout4(x)
        #x=self.pool5(x) 

        # AVG : [b, c=144 , 1]
        #x = F.avg_pool1d(x, x.shape[-1])  #reduces the remaining temporal dimension to a single value for each channel, summarizing the entire sequence.
        

        #c -> innput size et i -> seq len
        x = x.permute(2, 0, 1) # reshape  [15, b , 144]
       
        x = self.fc1(self.relu18(x)) # reduces redundancy and preparing the data for classification.
        
        # [15, b , 72]
        # Assuming self.xlstm is already defined
        hid_f = [l.init_hidden(x.size(1)) if isinstance(l, mLSTM) else l.init_hidden(x.size(1)) for l in self.xlstm_f]
        hid_b= [l.init_hidden(x.size(1)) if isinstance(l, mLSTM) else l.init_hidden(x.size(1)) for l in self.xlstm_f]


        # Pass the sequence through the mLSTM and sLSTM blocks
        out_f = []
        out_b = []

        # Compute model output and update the hidden states
        
        for seq in x:
            for i, lstm in enumerate(self.xlstm_f):

                out, hid_f[i] = lstm(seq, hid_f[i])

                out_f.append(out)



        x_rev = torch.flip(x, [0])
        for seq_rev in x_rev:
            for i, lstm in enumerate(self.xlstm_f):

                out_rev, hid_b[i] = lstm(seq_rev, hid_b[i])

                out_b.append(out_rev)

        #batch_first=True


        out_b.reverse()
        out_f=torch.stack(out_f, dim=1)[:,-1]
        out_b=torch.stack(out_b, dim=1)[:,-1]
        out = torch.cat((out_f,out_b), dim=1)
        
        return out
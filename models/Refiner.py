from torch import nn
import torch.nn.functional as F
from torch.utils import data

class RefinerV2(nn.Module):
    def __init__(self,L_in,horizon,filters=16,kernel=2,rm_window=10,conv_layers=2,transforms='all',neurons=600,block='conv'):
        super().__init__()

        self.trf=['cs','diff','devfm','fft','rm','acor']
        self.in_channels=1
        self.horizon=configs.pred_len
        self.out_channels=1
        self.lin_out=nn.Linear(self.horizon*(len(self.trf)+1),self.horizon)
        self.rm_window=configs.rm_window

        if type(kernel)==int:
            kernel_n=configs.kernel
            kernel=[]
            for n in range(conv_layers):
                    kernel.append(kernel_n)
        self.kernel=configs.kernel
        if type(neurons)==int:
            neurons_n=neurons
            neurons=[]
            for n in range(conv_layers):
                    neurons.append(neurons_n)
        if block=='conv':
            Block= Conv_Block
            self.init_block=[L_in,horizon,1,filters,kernel,conv_layers]
            self.init_block_rm=[L_in-rm_window+1,horizon,1,filters,kernel,conv_layers]
        elif block=='linear':
            Block= Linear_Block
            self.init_block=[L_in,horizon,neurons,conv_layers]
            self.init_block_rm=[L_in-rm_window+1,horizon,neurons,conv_layers]
        self.conv_layers=conv_layers
#-------------------------------------------------------------------------------------------------------
        #unaltered sequence block
        self.seq_block=Block(*self.init_block)
        #cumsum block
        if 'cs' in self.trf:
            self.cumsum_block=Block(*self.init_block)
        #diff block
        if 'diff' in self.trf:
            self.diff_block=Block(*self.init_block)
        #deviation form mean block
        if 'devfm' in self.trf:
            self.devfm_block=Block(*self.init_block)
        #real fft block
        if 'fft' in self.trf:
            self.rfft_block=Block(*self.init_block)
        #rolling mean block
        if 'rm' in self.trf:
            self.rm_block=Block(*self.init_block_rm)
        #acor block
        if 'acor' in self.trf:
            self.acor_block=Block(*self.init_block)

#-------------------------------------------------------------------------------------------------------

    def forward(self, x):
    #--------------------------------------------------
    #--------------------------------------------------
    #Blocks
    #--------------------------------------------------
    #--------------------------------------------------        
        #sequence block
        x_seq=self.seq_block(x)
        out=x_seq
        #-------------------------------------------------- 
        #cumsum block
        if 'cs' in self.trf:
            x_cs=torch.cumsum(x,dim=2)
            x_cs=self.cumsum_block(x_cs)
            out=torch.cat((out,x_cs),dim=1)
        #-------------------------------------------------- 
        #diff block
        if 'diff' in self.trf:
            x_dif=torch.diff(x,dim=2,prepend=torch.zeros([x.shape[0],x.shape[1],1],device=x.device.type))
            x_dif=self.diff_block(x_dif)
            out=torch.cat((out,x_dif),dim=1)
        #-------------------------------------------------- 
        #devfm block
        if 'devfm' in self.trf:
            x_dfm=torch.cumsum(x-x[:,:,0].reshape(-1,self.in_channels,1),dim=2)-torch.cumsum(x-x[:,:,0].reshape(-1,self.in_channels,1),dim=2)[:,:,-1].reshape(-1,self.in_channels,1)*torch.linspace(0,1,x.shape[2],device=x.device.type)
            x_dfm=self.devfm_block(x_dfm)
            out=torch.cat((out,x_dfm),dim=1)
        #-------------------------------------------------- 
        #rfft block
        if 'fft' in self.trf:
            x_rfft=torch.abs(torch.fft.fft(x))
            x_rfft=self.rfft_block(x_rfft)
            out=torch.cat((out,x_rfft),dim=1)
        #--------------------------------------------------
        #--------------------------------------------------  
        #rolling mean window 1
        if 'rm' in self.trf:
            x_rm=x.unfold(dimension=2, size=self.rm_window,step=1).mean(dim=3)    
            x_rm=self.rm_block(x_rm) 
            out=torch.cat((out,x_rm),dim=1)  
        #--------------------------------------------------
        #-------------------------------------------------- 
        #autocorrelation block
        if 'acor' in self.trf:
            x_acor=torch.flip(convolve(x,torch.flip(x,dims=[2]),mode='full'),dims=[2])[:,:,-x.shape[2]:]
            x_acor=self.acor_block(x_acor)
            out=torch.cat((out,x_acor),dim=1) 
    #--------------------------------------------------
    #--------------------------------------------------
    #Output
    #--------------------------------------------------
    #-------------------------------------------------- 
        out=self.lin_out(out)
        return out.reshape(-1,self.out_channels,self.horizon)


class Conv_Block(nn.Module):
    def __init__(self,L_in,horizon,input_channels,filters,kernel,conv_layers=2,act=nn.ReLU):
        super().__init__()
        layers=[nn.LayerNorm([L_in]),nn.Conv1d(input_channels,filters,kernel[0]),act()]
        for i in range(conv_layers-1):
            layers.append(nn.Conv1d(filters,filters,kernel[i+1]))
            layers.append(act())
        layers.append(nn.Flatten(start_dim=1))
        layers.append(nn.Linear((L_in+conv_layers-sum(kernel))*filters, horizon))
        self.layers=nn.Sequential(*layers)
    def forward(self,x):
        return(self.layers(x))

class Linear_Block(nn.Module):
    def __init__(self,L_in,horizon,neurons,conv_layers=2):
        super().__init__()
        layers=[nn.Flatten(start_dim=1),nn.Linear(L_in,neurons[0])]
        for i in range(conv_layers-1):
            layers.append(nn.Linear(neurons[i],neurons[i+1]))
            #layers.append(nn.ReLU())
        layers.append(nn.Linear(neurons[-1], horizon))
        self.layers=nn.Sequential(*layers)
    def forward(self,x):
        return(self.layers(x))
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import torch.nn.functional as F
import torch
from sklearn.preprocessing import StandardScaler
from models import LSTM, DLinear, PatchTST,Linear,TSMixer,PatchTST_c,Linear_c,TSMixer_c,MultiRocket

def clean_outliers(X,per=.95):
    '''X: weight matrix for a single index in the prediction dims= [B,L]'''
    percentiles = torch.quantile(torch.abs(X), per, dim=0)
    means = torch.mean(X, dim=0)
    mask = torch.abs(X) > percentiles
    X[mask] = means.expand_as(X)[mask]
    return X

def rescale_11(array):
    """Rescale "array" into the range [-1,1].
    
    Parameters
    _______________
    
    array: np.array
    Array to rescale
    
    """
    return (array-array.min()+array-array.max())/array.ptp()

def correlation_matrix(mat, arr):
    """Expand function corr2_coeff for several windows at the same time. Inputs would be GASF or GADF and the scores vector from either LRP or SHAP.
    
    Parameters
    _______________
    
    mat: np.array
    Array that contains several matrix windows. Shape (ner_windows, n, n)
    
    arr: np.array
    Array that contains several array windows. Shape (ner_windows, n)
    
    """
    
    if len(mat.shape) < 3: raise ValueError('Expected mat shape (ner_windows, n, n), provided {}'.format(mat.shape))
    if len(arr.shape) < 2: raise ValueError('Expected arr shape (ner_windows, n), provided {}'.format(arr.shape))
    
    if arr.shape[0] != mat.shape[0]:
        raise ValueError('mat and arr must have the same number of windows,'+
                         ' given {} (mat) and {} (arr)'.format(mat.shape[0], arr.shape[0]))
    
    corr_matrix = np.empty((arr.shape[0], arr.shape[1],1))
    
    for i in range(arr.shape[0]):
    
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = mat[i] - mat[i].mean(1)[:, None]
        B_mB = arr[i, None] - arr[i, None].mean(1)[:, None]
        
        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)


        # Finally get corr coeff
        corr_matrix[i] = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
        
    return corr_matrix

def create_dataset(dataset, lookback,lookfront=1,backend='torch'):

    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the channel, second is the batch_size
        lookback: Size of window for prediction
        lookfront: Size of the output. Number of future time steps 
        backend: torch or numpy
    """
    X= np.zeros(shape=[(dataset.shape[1]-lookback-lookfront),dataset.shape[0],lookback])
    y=np.zeros(shape=[(dataset.shape[1]-lookback-lookfront),dataset.shape[0],lookfront])
    for i in range(dataset.shape[1]-lookback-lookfront):
        feature = dataset[:,i:i+lookback]
        target = dataset[:,i+lookback:i+lookback+lookfront]
        X[i]=feature
        y[i]=target
    if backend=='torch':
        return torch.tensor(X,dtype=torch.float), torch.tensor(y,dtype=torch.float)
    else:
        return X, y

def load_euma(lookback,horizon,**kwargs):
    df=pl.read_csv('./dataset/euma.csv')
    Y_df=df.select(
        ((pl.col('vol_up')+pl.col('vol_dn')).alias('y')),
        ((pl.col('date_time').str.to_datetime("%Y-%m-%d %H:%M:%S")).alias('ds')),
        (pl.col('new_name').alias('unique_id'))
    )
    Y_df=Y_df.sort('ds')
    Y_df=Y_df.groupby_dynamic('ds',every='10m').agg(pl.col('y').sum(),pl.lit(1).alias('unique_id'))  #Aggregate all of the applications
    Y_df=Y_df[:-1200].to_pandas()
    n_time = len(Y_df['ds'].unique())
    val_size = int(.2 * n_time)
    test_size = int(.2 * n_time)
    scaler=StandardScaler()
    scaler.fit(np.array(Y_df['y'].iloc[:-test_size]).reshape(-1,1))
    Y_df['y']=scaler.transform(np.array(Y_df['y']).reshape(-1,1)).squeeze()
    n_series = len(Y_df.unique_id.unique())
    time_series=np.array(Y_df['y']).reshape(n_series, -1)
    X,y=create_dataset(time_series,lookback,horizon)
    X,y=X.permute(0,2,1),y.permute(0,2,1)
    X_train,y_train=X[:-test_size],y[:-test_size]
    X_test,y_test=X[-test_size:],y[-test_size:]
    return X_train,X_test,y_train,y_test

def load_users(lookback,horizon,frequency=3,**kwargs):
    df=pl.read_csv('./dataset/users_allBS.csv')
    df=df.filter(pl.col('frequency')==frequency)

    Y_df=df.select(
        (pl.col('user_unique').alias('y')),
        (pl.from_epoch("timestamp", time_unit="s").alias('ds')),
        (pl.col('frequency').alias('unique_id'))
    )
    Y_df=Y_df.sort('ds')
    Y_df=Y_df.groupby_dynamic('ds',every='10m').agg(pl.col('y').sum(),pl.lit(1).alias('unique_id'))
    Y_df=Y_df.to_pandas()
    n_time = len(Y_df['ds'].unique())
    val_size = int(.1 * n_time)
    test_size = int(.2 * n_time)
    scaler=StandardScaler()
    scaler.fit(np.array(Y_df['y'].iloc[:-test_size]).reshape(-1,1))
    Y_df['y']=scaler.transform(np.array(Y_df['y']).reshape(-1,1)).squeeze()
    n_series = len(Y_df.unique_id.unique())
    time_series=np.array(Y_df['y']).reshape(n_series, -1)
    X,y=create_dataset(time_series,lookback,horizon)
    X,y=X.permute(0,2,1),y.permute(0,2,1)
    X_train,y_train=X[:-test_size],y[:-test_size]
    X_test,y_test=X[-test_size:],y[-test_size:]
    return X_train,X_test,y_train,y_test

def load_UCR(lookback,horizon,dataset,**kwargs):
    data_dict={'earthquakes':'Earthquakes','italypd':'ItalyPowerDemand','chinatown':'Chinatown','plane':'Plane','yoga':'Yoga','eumaclf':'EUMA','computers':'Computers'}
    name=data_dict[dataset]
    train,test=np.loadtxt(f'./dataset/{name}/{name}_TRAIN.tsv',delimiter='\t'),np.loadtxt(f'./dataset/{name}/{name}_TEST.tsv',delimiter='\t')
    scaler=StandardScaler()
    scaler.fit(train[:,1:])
    X_train,y_train=torch.tensor(scaler.transform(train[:,1:])).float(),torch.tensor(train[:,0]).long()
    X_test,y_test=torch.tensor(scaler.transform(test[:,1:])).float(),torch.tensor(test[:,0]).long()
    if y_train.min()>0:
        y_train=y_train-1
        y_test=y_test-1
    return X_train.unsqueeze(dim=-1),X_test.unsqueeze(dim=-1),y_train,y_test

def load_basic(lookback, horizon, dataset, scaler=StandardScaler(), split=[0.8, 0.0, 0.2] ,univariate= 'True',**kwargs):
    '''Generate tensors for training, testing, and validating.
        dataset: path to dataset
        lookback: window size
        horizon: prediction length
        scaler: scaling class to use
        split: train-val-test split
    '''
    dataset_dict={'ili': './dataset/national_illness.csv','etth2':'./dataset/ETTh2.csv','Vtraffic':'./dataset/Vtraffic.csv','synthetic':'./dataset/synthetic.csv','cperiod':'./dataset/cperiod.csv'}
    dataset=dataset_dict[dataset]
    # Check if split size is 3
    if len(split) != 3:
        raise ValueError("Size of train-val-test split must be 3")
    
    # Check if the sum of split elements is equal to 1
    if sum(split) != 1:
        raise ValueError("The sum of train-val-test elements must be equal to 1")
    
    data = pd.read_csv(dataset).values[:, 1:]
    train_size, test_size = int(split[0] * data.shape[0]), int(split[2] * data.shape[0])
    val_size = len(data)-train_size - test_size
    scaler.fit(data[:-test_size])  # ensure to scale values excluding the test set
    data_s = scaler.transform(data)
    if univariate=='True':
        data_s=data_s[:,-1:] #make it univariate
    else: 
        pass
    X_train,y_train=create_dataset(np.swapaxes(data_s, 1, 0)[:,:train_size], lookback, horizon)
    X_test,y_test=create_dataset(np.swapaxes(data_s, 1, 0)[:,-test_size-lookback:], lookback, horizon)
    return X_train.permute(0,2,1),X_test.permute(0,2,1),y_train.permute(0,2,1),y_test.permute(0,2,1)

class Configuration():
    def __init__(self,seq_len,pred_len):
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.enc_in=1
        self.e_layers=3
        self.n_heads=4
        self.d_model=16
        self.d_ff=128
        self.dropout=0.3
        self.fc_dropout=0.3
        self.head_dropout=0
        self.patch_len=24
        self.stride=2
        self.individual=0
        self.padding_patch='end'
        self.revin=1
        self.affine=0
        self.subtract_last=0
        self.decomposition=0
        self.kernel_size=25

def err_h(pred,y,variate='all'):
    '''Error horizon'''
    error_mse=[]
    error_mae=[]
    if variate !='all':
        v=variate
        for i in range(pred.shape[1]):
            error_mse.append(F.mse_loss(pred[:,i,v],y[:,i,v]).item())
            error_mae.append(F.l1_loss(pred[:,i,v],y[:,i,v]).item())
    else:
        for i in range(pred.shape[1]):
            error_mse.append(F.mse_loss(pred[:,i],y[:,i]).item())
            error_mae.append(F.l1_loss(pred[:,i],y[:,i]).item())

    return error_mse,error_mae

def err_f(pred,y,variate='all'):
    error_mse=[]
    error_mae=[]
    if variate!='all':
        v=variate
        for i in range(pred.shape[0]):
            error_mse.append(F.mse_loss(pred[i,:,v],y[i,:,v]).item())
            error_mae.append(F.l1_loss(pred[i,:,v],y[i,:,v]).item())
    else:
        for i in range(pred.shape[0]):
            error_mse.append(F.mse_loss(pred[i],y[i]).item())
            error_mae.append(F.l1_loss(pred[i],y[i,]).item())

    return error_mse,error_mae

def naive_predictor(X,horizon):
    '''Naive predictor'''
    pred=torch.zeros((X.shape[0],horizon,1))
    pred[:,:,:]=X[:,-1:,:]
    return pred


def random_subset(tensor, num_elements):
    """
    Selects a random subset from the given tensor.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    num_elements (int): The number of elements in the subset.

    Returns:
    torch.Tensor: A tensor containing the random subset.
    """
    # Check if the number of elements requested is more than the number of elements in the tensor
    if num_elements > tensor.size(0):
        raise ValueError("Number of elements requested exceeds the number of elements in the tensor.")
    
    # Generate random indices
    random_indices = torch.randperm(tensor.size(0))[:num_elements]
    
    # Select the elements using the random indices
    subset = tensor[random_indices]
    
    return subset

def w_diff(classes,w_matrix):
    return w_matrix[:,:,classes[0]]-w_matrix[:,:,classes[1]]

def linear_xai(X,s):
    '''
    X: time series windows. [B,LxC]
    s: shap values. [B,LxC,h]
    '''
    X=X.unsqueeze(dim=-1)
    s=torch.tensor(s)
    s_x=s/X
    s_x[torch.isinf(s_x)]=0
    xai_matrix=s_x[1:]-s_x[:-1]
    return s_x,torch.cat((torch.zeros_like(xai_matrix)[:1], xai_matrix), dim=0)

dataset_dict={'users':load_users,'euma':load_euma,'ili':load_basic,
'etth2':load_basic,'Vtraffic':load_basic,'synthetic':load_basic,
'earthquakes':load_UCR,'italypd':load_UCR,'chinatown':load_UCR,'plane':load_UCR,'yoga':load_UCR,'eumaclf':load_UCR,
'computers':load_UCR} #add here your loading function for your dataset

model_dict={'lstm': LSTM.Model,'patchtst': PatchTST.Model,'dlinear': DLinear.Model,'linear':Linear.Model,'tsmixer':TSMixer.Model,'linear_c':Linear_c.Model,
'tsmixer_c':TSMixer_c.Model,'patchtst_c': PatchTST_c.Model,'multirocket':MultiRocket.Model} 
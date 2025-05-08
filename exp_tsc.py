import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import shap
import polars
import os
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models import LSTM, DLinear, PatchTST, Linear, TSMixer, MultiRocket
from models import layers
from utils import *
import pickle
from pyts.image import GramianAngularField
import copy


parser = argparse.ArgumentParser(description='AIChronoLens')
#Experiment setup
parser.add_argument('--dataset', type=str, default='users', help='dataset')
parser.add_argument('--users_bs', type=int, default=3, help='BS of users dataset (3,4 or 5)')
parser.add_argument('--model', type=str, required=True, default='lstm', help='model name, default options: [lstm, patchtst, dlinear]')
parser.add_argument('--save_model', type=bool, default=True, help='save a copy of the models state dict')   
parser.add_argument('--seq_len', type=int, default=300, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=60, help='prediction sequence length')
parser.add_argument('--univariate', type=str, default='True', help='True for making the time series univariate')
parser.add_argument('--save_train_info', type=str, default='False', help='saving shap values and corr matrix for training set')
parser.add_argument('--shap_samples', default='all', help='samples to use in train set shap calculation')
parser.add_argument('--background_samples',type=int, default=100, help='background samples used in shap calculation')
#SHAP
parser.add_argument('--calculate_shap', type=str, default='True', help='If true shap values are calculated')
#parser.add_argument('--horizon_item', type=int, default=59, help='horizon index to be explained by shap')
parser.add_argument('--max_evals', type=int, default=620, help='maximum number of evaluations during shap values calculation')
parser.add_argument('--shap_batch_size', type=int, default=1600, help='shap batch size for shap calculation')
# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# NeuroFLexMLP
parser.add_argument('--layers', type=int, default=1, help='number of residual blocks')
parser.add_argument('--linear_n', type=int, default=1600, help='hidden size in linear layer')
parser.add_argument('--nonlinear_n', type=int, default=400, help='hidden size in non-linear layer')

#TSMixer
parser.add_argument('--feat_mixing_hidden_channels', type=int, default=6, help='feat_mixing_hidden_channels')
parser.add_argument('--no_mixer_layers', type=int, default=6, help='no_mixer_layers')
parser.add_argument('--tsmixer_eps', type=float, default=1e-8, help='eps for numerical stability')

#optimization
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=25, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--patience_scheduler', type=int, default=3,help='patience for ReduceLROnPlateau scheduler') 

args = parser.parse_args()
if args.dataset=='users':
    with open(f'./results/configs/{args.model}_{args.dataset}_bs{args.users_bs}_{args.seq_len}_{args.pred_len}_configs.pkl', 'wb') as f:
            pickle.dump(args, f)
else:
    with open(f'./results/configs/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}_configs.pkl', 'wb') as f:
            pickle.dump(args, f)

#-------------------------------------------------------------------------------------------------------------------------
X_train,X_test,y_train,y_test=dataset_dict[args.dataset](args.seq_len,args.pred_len,dataset=args.dataset,frequency=args.users_bs,univariate=args.univariate)
if args.dataset=='users':
    args.dataset=f'users_bs{args.users_bs}'
print(X_train.shape,X_test.shape)
#Model training
model = model_dict[args.model](args)
if args.univariate!='True':
    args.model=args.model+'_mv'
model.to(dev)
with open(f'./results/configs/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}_configs.pkl', 'wb') as f:
        pickle.dump(args, f)
loss_fn = torch.nn.CrossEntropyLoss()

train_loader = data.DataLoader(data.TensorDataset(X_train[:int(X_train.shape[0]*0.8)], y_train[:int(y_train.shape[0]*0.8)]), shuffle=True, batch_size=args.batch_size)
val_loader=data.DataLoader(data.TensorDataset(X_train[int(X_train.shape[0]*0.8):], y_train[int(y_train.shape[0]*0.8):]), shuffle=False, batch_size=args.batch_size)
test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=False, batch_size=args.batch_size)
optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience_scheduler,threshold=5e-3)
if args.model=='multirocket':
    model.fit(X_train.detach()[:,:,0].numpy().astype(np.float32),y_train.detach().numpy().astype(np.float32))
last_lr=args.learning_rate
min_loss=float("inf")
for ep in range(args.epochs):
    model.train()
    loss_train=0
    for i,(X,y) in enumerate(train_loader):
        X,y=X.to(dev),y.to(dev)
        loss = loss_fn(model(X).squeeze(), y)
        loss_train+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del X 
        del y
    loss_train/= len(train_loader)
    model.eval()  
    with torch.no_grad():
        val_loss=0
        for X, y in val_loader:
            X,y=X.to(dev),y.to(dev)
            val_pred = model(X).squeeze()
            val_loss += loss_fn(val_pred, y)
            del X
            del y
    val_loss /= len(val_loader) 
    if val_loss<min_loss:
        print(f'new best validation loss: {val_loss}')
        best_state_dict=model.state_dict()
        torch.save(best_state_dict, f'./results/trained_models/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}.pt')
        best_model=copy.deepcopy(model)
        min_loss=val_loss
    with torch.no_grad():
        test_loss=0
        for X, y in test_loader:
            X,y=X.to(dev),y.to(dev)
            test_pred = model(X).squeeze()
            test_loss += loss_fn(test_pred, y)
            del X
            del y
    test_loss /= len(test_loader)  
    scheduler.step(loss_train)        
    print("Epoch %d: train loss %.6f, val loss %.6f,  test loss %.6f " % (ep, loss_train,val_loss, test_loss))
best_model.cpu()
best_model.eval()
#model.load_state_dict(best_state_dict)
predictions=best_model(X_test).squeeze().cpu().detach().numpy()
if len(predictions.shape)>2:
    np.savetxt(f'./results/predictions/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}_predictions.txt', predictions.reshape(-1,predictions.shape[1]*predictions.shape[2]))
else:
    np.savetxt(f'./results/predictions/{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}_predictions.txt', predictions)

#SHAP
if args.calculate_shap == 'True':
    best_model.to(dev)
    def model_for_shap(X):
        X=torch.tensor(X).float().reshape((-1,args.seq_len, args.enc_in))
        if args.shap_batch_size>len(X):
            args.shap_batch_size=len(X)
        loader = data.DataLoader(X,batch_size=args.shap_batch_size)
        print(X.shape)
        preds=torch.tensor([])
        with torch.no_grad():
            for X in loader:
                X=X.to(dev)
                if len(X)!=1:
                    pred = best_model(X).cpu()
                else:
                    pred=best_model(X.expand(2,-1,-1)).cpu()[0:1,:,:] #numba error when X is == 1
                del X
                preds=torch.cat((preds,pred),dim=0)
        return preds.detach().reshape(-1,preds.shape[1]*preds.shape[2]).numpy()

    size=args.background_samples
    train_array=X_train.reshape(X_train.shape[0],args.seq_len*args.enc_in).cpu().detach().numpy()[np.random.choice(X_train.shape[0], size=size, replace=False)]
    test_array=X_test.reshape(X_test.shape[0],args.seq_len*args.enc_in).cpu().detach().numpy()
    explainer = shap.KernelExplainer(model_for_shap,train_array)
    shap_values = explainer(test_array)
    with open(f'./results/shap_values/shap_values_test_{args.dataset}_{args.model}_{args.seq_len}_{args.pred_len}.pkl', 'wb') as file:
            pickle.dump(shap_values, file)
    if args.save_train_info == 'True':
        if args.shap_samples!='all':
            shap_values_train = explainer(X_train[-args.shap_samples:].reshape(X_train.shape[0],args.seq_len*args.enc_in).cpu().detach().numpy())
        else:
            shap_values_train = explainer(X_train.reshape(X_train.shape[0],args.seq_len*args.enc_in).cpu().detach().numpy())
        with open(f'./results/shap_values/shap_values_train_{args.dataset}_{args.model}_{args.seq_len}_{args.pred_len}.pkl', 'wb') as file:
                pickle.dump(shap_values_train, file)







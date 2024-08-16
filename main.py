import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import os
import logging
from data import NPZDataset
import numpy as np
from utils import Metric,get_model_size,test_speed,beijing_time, set_logger,init_weights,set_seed
import argparse
from models.KANFormer import KANFormer


parse = argparse.ArgumentParser()

parse.add_argument('--model', type=str,default='KANFormer')
parse.add_argument('--log_out', type=int,default=0)
parse.add_argument('--dataset', type=str,default='CAVE')
parse.add_argument('--check_point', type=str,default=None)
parse.add_argument('--check_step', type=int,default=50)
parse.add_argument('--lr', type=int, default=4e-4)
parse.add_argument('--batch_size', type=int, default=32)
parse.add_argument('--epochs', type=int,default=1000)
parse.add_argument('--seed', type=int,default=3407) 
parse.add_argument('--scale', type=int,default=4)
parse.add_argument('--hidden_dim', type=int,default=256)
parse.add_argument('--depth', type=int,default=4)
parse.add_argument('--comments', type=str,default='')
parse.add_argument('--grid_size', type=int,default=5)
parse.add_argument('--spline_order', type=int,default=3)

args = parse.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = args.model
model = None
HSI_bands = 31


test_dataset_path = None
train_dataset_path = None
if args.dataset == 'CAVE':
    HSI_bands = 31
    if args.scale == 2:
        test_dataset_path = './datasets/CAVE_test_x2.npz'
        train_dataset_path = './datasets/CAVE_train_x2.npz'
    if args.scale == 4:
        test_dataset_path = './datasets/CAVE_test_x4.npz'
        train_dataset_path = './datasets/CAVE_train_x4.npz'
    if args.scale == 8:
        test_dataset_path = './datasets/CAVE_test_x8.npz'
        train_dataset_path = './datasets/CAVE_train_x8.npz'

if args.model == "KANFormer":
    model = KANFormer(scale=args.scale,depth=args.depth)


model = model.to(device)
set_seed(args.seed)
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.Adam(lr=args.lr,params=model.parameters())
scheduler = StepLR(optimizer=optimizer,step_size=100,gamma=0.1)
# test_dataset = NPZDataset(test_dataset_path)
train_dataset = NPZDataset(train_dataset_path)
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
# test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size * 4)
start_epoch = 0 
inference_time,flops,params = test_speed(model,device,HSI_bands,scale=args.scale)
if args.check_point is not None:
    checkpoint = torch.load(args.check_point)  
    model.load_state_dict(checkpoint['net'],strict=False)  
    optimizer.load_state_dict(checkpoint['optimizer']) 
    start_epoch = checkpoint['epoch']+1 
    scheduler.load_state_dict(checkpoint['scheduler'])
    log_dir,_ = os.path.split(args.check_point)
    
    print(f'check_point: {args.check_point}')
    
if args.check_point is  None:
    init_weights(model)
    log_dir = f'./trained_models/{model_name}_x{args.scale}_{args.comments},{args.dataset},{beijing_time()}'
    if not os.path.exists(log_dir) and args.log_out == 1:
        os.mkdir(log_dir)

logger = set_logger(model_name, log_dir, args.log_out)
model_size = get_model_size(model)
logger.info(f'[model:{model_name}_x{args.scale}({args.comments}),dataset:{args.dataset}],model_size:{params}M,inference_time:{inference_time:.6f}S,FLOPs:{flops}G')

import time
def train():
    model.train()
    for epoch in range(start_epoch, args.epochs):
        loss_list = []
        start_time = time.time()
        for idx,loader_data in enumerate(train_dataloader):
            GT,LRHSI,RGB = loader_data[0].to(device),loader_data[1].to(device),loader_data[2].to(device)
            preHSI = model(LRHSI,RGB)
            loss = loss_func(GT,preHSI)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        logging.info(f'Epoch:{epoch},loss:{np.mean(loss_list)},time:{time.time()-start_time:.2f}s')
        
if __name__ == "__main__":
    train()
    
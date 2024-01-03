import torch
from torch.utils.data import DataLoader
import configparser
import argparse
from time import time
import sys
import os
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
import math

from pile_dataset import pile_dataset
from model import Model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--resume_training_starting_with_epoch', type=int, default=0)
parser.add_argument('--resume_training_starting_with_chunk_num', type=int, default=0)
args = parser.parse_args()

resume_training_starting_with_epoch=args.resume_training_starting_with_epoch
resume_training_starting_with_chunk_num=args.resume_training_starting_with_chunk_num

config=configparser.ConfigParser()
config.read("config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')

device=config.get('train', 'device')
epochs=int(config.get('train', 'epochs'))
num_of_chunks_to_use=int(config.get('train', 'num_of_chunks_to_use'))
num_workers=int(config.get('train', 'num_workers'))
prefetch_factor=int(config.get('train', 'prefetch_factor'))
max_lr=float(config.get('train', 'max_lr'))
start_predicting_from_what_token=int(config.get('train', 'start_predicting_from_what_token'))
P_drop=float(config.get('train', 'P_drop'))
init_std=float(config.get('train', 'init_std'))
masking_minus_inf=float(config.get('train', 'masking_minus_inf'))
if(config.get('train', 'disable_pin_memory')=="True"):
    pin_memory=False
else:
    pin_memory=True
folder_to_save_state_dicts=config.get('train', 'folder_to_save_state_dicts')
warmup_steps=int(config.get('train', 'warmup_steps'))
steps_to_attenuate_lr_to_zero=int(config.get('train', 'steps_to_attenuate_lr_to_zero'))
weight_decay=float(config.get('train', 'weight_decay'))

N=int(config.get('architecture', 'N'))
d_model=int(config.get('architecture', 'd_model'))
d_ff=int(config.get('architecture', 'd_ff'))
h=int(config.get('architecture', 'h'))
d_k=int(config.get('architecture', 'd_k'))
d_v=int(config.get('architecture', 'd_v'))
positional_encoding_max_pos=int(config.get('architecture', 'positional_encoding_max_pos'))

tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(path_to_store_preprocessed_data,"tokenizer"))
vocab_size=len(tokenizer)

writer = SummaryWriter('tensorboard/run')

folder_exists = os.path.isdir(folder_to_save_state_dicts)
if(not folder_exists):
    os.makedirs(folder_to_save_state_dicts)

model = Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, init_std, positional_encoding_max_pos, masking_minus_inf, device).cuda(device)  

schedule_fn = lambda step_num: min(step_num/warmup_steps, math.cos(math.pi*(step_num-warmup_steps)/(2*steps_to_attenuate_lr_to_zero)))


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn, verbose=False)

if(resume_training_starting_with_chunk_num!=0):
    checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(resume_training_starting_with_epoch) + '_c' + str(resume_training_starting_with_chunk_num-1) +'.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    del checkpoint
elif(resume_training_starting_with_epoch!=0 and resume_training_starting_with_chunk_num==0):
    checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(resume_training_starting_with_epoch-1) + '_c' + str(num_of_chunks_to_use-1) +'.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])    
    del checkpoint

for epoch in range(resume_training_starting_with_epoch, epochs):
    for chunk_num in range(resume_training_starting_with_chunk_num, num_of_chunks_to_use):

        training_data=pile_dataset(chunk_num)
                
        dataloader = DataLoader(training_data, batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, prefetch_factor=prefetch_factor)
        
        num_of_batches=len(dataloader.dataset)

        s=time()
        
        print(f"\n\n\nEpoch: {epoch}, chunk_num: {chunk_num}\n")
        
        
        total_loss=0
        for batch_num, (batch_input_onehot, batch_output_int) in enumerate(dataloader):
            
            utils.display_progress(batch_num, num_of_batches)
            
            optimizer.zero_grad()
            
            batch_output_int=batch_output_int.to(device)    
            batch_input_onehot=batch_input_onehot.to(device)
            
            
            
            batch_output_pred = model(batch_input_onehot)                   
            loss=loss_fn(batch_output_pred[:,:,start_predicting_from_what_token:], batch_output_int[:,start_predicting_from_what_token:])
            
            
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            total_loss+=loss.cpu().detach().numpy()
            
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }, folder_to_save_state_dicts + '/state_dict_e' + str(epoch) + '_c' + str(chunk_num) + '.pt')
            
        e=time()
        
        print("\nChunk running time: {0:4.1f} hours".format((e-s)/3600))
        print("Loss: {0:4.4f}".format(total_loss/num_of_batches))
        
        writer.add_scalar("Loss/train", total_loss/num_of_batches, epoch*num_of_chunks_to_use + chunk_num)
        writer.flush()
        
writer.close()
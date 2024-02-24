import torch
import torch.nn.functional as F
import configparser
# import sys
import os
from transformers import GPT2Tokenizer
import argparse
import numpy as np

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', required=True)
args = parser.parse_args()


config=configparser.ConfigParser()
config.read("config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')

device=config.get('inference', 'device')
how_many_tokens_to_generate=int(config.get('inference', 'how_many_tokens_to_generate'))
epochs=int(config.get('train', 'epochs'))
chunk_num=int(config.get('inference', 'chunk_num'))

epoch=epochs-1

P_drop=float(config.get('train', 'P_drop'))
init_std=float(config.get('train', 'init_std'))
masking_minus_inf=float(config.get('train', 'masking_minus_inf'))

folder_to_save_state_dicts=config.get('train', 'folder_to_save_state_dicts')


N=int(config.get('architecture', 'N'))
d_model=int(config.get('architecture', 'd_model'))
d_ff=int(config.get('architecture', 'd_ff'))
h=int(config.get('architecture', 'h'))
d_k=int(config.get('architecture', 'd_k'))
d_v=int(config.get('architecture', 'd_v'))
positional_encoding_max_pos=int(config.get('architecture', 'positional_encoding_max_pos'))

tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(path_to_store_preprocessed_data,"tokenizer"))
vocab_size=len(tokenizer)

ids_list = tokenizer.encode(args.prompt)
bos_id=tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

ids_list = [bos_id] + ids_list

for k in range(how_many_tokens_to_generate):
        
    prompt_onehot_np=np.zeros((1, len(ids_list), vocab_size), dtype=np.float32)
    
    for i in range(len(ids_list)):
        prompt_onehot_np[0,i,ids_list[i]]=1
        
    prompt_onehot=torch.from_numpy(prompt_onehot_np).to(device)
    
    
    model = Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, init_std, positional_encoding_max_pos, masking_minus_inf, device).cuda(device)
    
    
    checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(epoch) + '_c' + str(chunk_num) +'.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    
    output_pred = model(prompt_onehot)
    
    pred_id = torch.argmax(F.softmax(torch.squeeze(output_pred, dim=0), dim=0), dim=0).detach().cpu().numpy()[-1]
    
    ids_list.append(pred_id)
    
final=tokenizer.decode(ids_list, skip_special_tokens = True)

print(final)




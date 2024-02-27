import torch
import configparser
import sys
import os
from transformers import GPT2Tokenizer
import argparse
from utils import inference_greedy_search, inference_beam_search

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
beam_size=int(config.get('inference', 'beam_size'))
inference_method=config.get('inference', 'inference_method')

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

model = Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, init_std, positional_encoding_max_pos,
              masking_minus_inf, device).cuda(device)

checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(epoch) + '_c' + str(chunk_num) + '.pt',
                        map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

if(inference_method=="greedy_search"):
    ids_list_output = inference_greedy_search(model, ids_list, vocab_size, device, how_many_tokens_to_generate)
elif(inference_method=="beam_search"):
    ids_list_output = inference_beam_search(model, ids_list, vocab_size, device, how_many_tokens_to_generate, beam_size)
else:
    sys.exit("Please, specify a valid inference_method in [inference] section of config file. Valid values are greedy_search and beam_search.")

final = tokenizer.decode(ids_list_output, skip_special_tokens=True)

print(final)




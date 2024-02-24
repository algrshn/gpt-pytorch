import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import configparser
import sys
import os
from transformers import GPT2Tokenizer
from math import log2

from pile_dataset import pile_dataset
from model import Model
import utils



config=configparser.ConfigParser()
config.read("config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')

device=config.get('inference', 'device')
epochs=int(config.get('train', 'epochs'))
chunk_num=int(config.get('inference', 'chunk_num'))
chunk_num_val=int(config.get('val', 'chunk_num_val'))
num_of_docs=int(config.get('val', 'num_of_docs'))
num_workers=int(config.get('train', 'num_workers'))
prefetch_factor=int(config.get('train', 'prefetch_factor'))
start_predicting_from_what_token=int(config.get('train', 'start_predicting_from_what_token'))
P_drop=float(config.get('train', 'P_drop'))
init_std=float(config.get('train', 'init_std'))
masking_minus_inf=float(config.get('train', 'masking_minus_inf'))
if(config.get('train', 'disable_pin_memory')=="True"):
    pin_memory=False
else:
    pin_memory=True
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

model = Model(vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, init_std, positional_encoding_max_pos,
              masking_minus_inf, device).cuda(device)


checkpoint = torch.load(folder_to_save_state_dicts + '/state_dict_e' + str(epochs-1) + '_c' + str(chunk_num) +'.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint

model.eval()

val_data=pile_dataset(chunk_num_val)

dataloader = DataLoader(val_data, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, prefetch_factor=prefetch_factor)

num_of_batches=len(dataloader.dataset)

log_doc_perplexities=[]
for batch_num, (batch_input_onehot, batch_output_int) in enumerate(dataloader):

    utils.display_progress(batch_num, num_of_batches)

    batch_output_int=batch_output_int.to(device)
    batch_input_onehot=batch_input_onehot.to(device)

    batch_onehot_pred = model(batch_input_onehot)
    batch_prob = F.softmax(batch_onehot_pred, dim=1).cpu().detach().numpy()
    batch_input = torch.argmax(batch_input_onehot, dim=2).cpu().detach().numpy()


    for i in range(batch_prob.shape[0]):
        doc_prob=batch_prob[i,:,:]
        doc_input=batch_input[i,:]

        log_perplexity_not_normalized = 0
        for j in range(start_predicting_from_what_token, doc_input.shape[0]-1):
            word_id=int(doc_input[j+1])
            word_prob=doc_prob[word_id, j]
            log_perplexity_not_normalized+=-log2(word_prob)

        log_doc_perplexity = log_perplexity_not_normalized / (doc_input.shape[0] - 1 - start_predicting_from_what_token)
        log_doc_perplexities.append(log_doc_perplexity)


total_perplexity=2**(sum(log_doc_perplexities)/len(log_doc_perplexities))

print("\n\n\nPerplexity after training on {0} chunks: {1:4.2f}".format(chunk_num+1,total_perplexity))



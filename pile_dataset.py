from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import numpy as np
import pickle
import configparser
import os

config=configparser.ConfigParser()
config.read("config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')

tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(path_to_store_preprocessed_data,"tokenizer"))
bos_id=tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
eos_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
vocab_size=len(tokenizer)

batched_folder=os.path.join(path_to_store_preprocessed_data,"batched")


def load_chunk(chunk_num):
    
    with open(os.path.join(batched_folder, str(chunk_num) + "_batched.pickle"), "rb") as f:
        chunk=pickle.load(f)
        
    return chunk


class pile_dataset(Dataset):
    def __init__(self, chunk_num):
        self.data = load_chunk(chunk_num)
        self.num_of_batches = len(self.data)

    def __len__(self):
                                    
        return self.num_of_batches

    def __getitem__(self, idx):
        
        batch=self.data[idx]
        batch_size=batch.shape[0]
        
        eos_column=np.empty((batch_size,1), dtype=np.int64)
        eos_column.fill(eos_id)
        batch_output_int=np.concatenate((batch, eos_column), axis=1)
        
        bos_column=np.empty((batch_size,1), dtype=np.int64)
        bos_column.fill(bos_id)
        batch_input_int=np.concatenate((bos_column, batch), axis=1)
        
        batch_input_onehot=np.zeros((batch_size, batch_input_int.shape[1], vocab_size), dtype=np.float32)
        
        for i in range(batch_size):            
            for pos in range(batch_input_int.shape[1]):
                batch_input_onehot[i, pos, batch_input_int[i,pos]]=1
            
        return (batch_input_onehot, batch_output_int)
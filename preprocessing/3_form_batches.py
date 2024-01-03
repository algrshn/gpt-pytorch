import configparser
import json
import argparse
import numpy as np
import pickle
import os

import utils


# parser = argparse.ArgumentParser()
# parser.add_argument('--chunk_num', type=int, required=True)
# args = parser.parse_args()

# chunk_num = args.chunk_num

config=configparser.ConfigParser()
config.read("../config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')
max_len=int(config.get('preprocessing', 'max_len_tokens'))
max_num_of_tokens_in_batch=int(config.get('preprocessing', 'max_num_of_tokens_in_batch'))

for chunk_num in range(3,30):
    
    print(f"chunk_num={chunk_num}")
    
    print("Loading data ...")
    with open(os.path.join(path_to_store_preprocessed_data, "tokenized", str(chunk_num) + "_tokenized.json"), "r") as f:
        tokenized = json.load(f)
    print("Done\n\n")
    
    print("Truncating ...")    
    tokenized_truncated=[]
    for text in tokenized:
        if(len(text)>max_len):
            text_truncated=text[:max_len]
            tokenized_truncated.append(text_truncated)
        else:
            tokenized_truncated.append(text)
    print("Done\n\n")
    
    print("Sorting ...")
    len_list=[]
    for text in tokenized_truncated:
        len_list.append(len(text))
    
    zipped = sorted(zip(len_list,tokenized_truncated), key = lambda x: x[0])
    len_list,tokenized_truncated = zip(*zipped)
    print("Done\n\n")
    
    print("Splitting into batches ...")
    catalog_of_len=list(set(len_list))
    num_of_texts_with_that_length=[]
    for length in catalog_of_len:
        num=len_list.count(length)
        num_of_texts_with_that_length.append(num)
    
    output=[]
    start_pos=0
    for k in range(len(catalog_of_len)):
        
        length=catalog_of_len[k]+1
        num=num_of_texts_with_that_length[k]
        regular_batch_size=max_num_of_tokens_in_batch//length
        num_of_full_size_batches=num//regular_batch_size
        truncated_batch_size=num - num_of_full_size_batches*regular_batch_size
        
        for n in range(num_of_full_size_batches):
            batch=[]
            for i in range(start_pos, start_pos+regular_batch_size):
                batch.append(tokenized_truncated[i])
        
            start_pos += regular_batch_size
            np_batch=np.array(batch, dtype=int)    
            output.append(np_batch) 
            
            utils.display_progress(start_pos, len(tokenized_truncated))
            
            
        if(truncated_batch_size>0):
            batch=[]
            for i in range(start_pos, start_pos+truncated_batch_size):
                batch.append(tokenized_truncated[i])
        
            start_pos += truncated_batch_size
            np_batch=np.array(batch, dtype=int)    
            output.append(np_batch)   
           
    print("\nDone\n\n")    
    
    print("Saving the results ...")
    
    batched_folder=os.path.join(path_to_store_preprocessed_data,"batched")
    if not os.path.exists(batched_folder):
        os.makedirs(batched_folder)
    
    with open(os.path.join(batched_folder, str(chunk_num) + "_batched.pickle"), "wb") as f:
        pickle.dump(output, f)
    
    print("Done")
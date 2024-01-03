import json
from transformers import GPT2Tokenizer
import configparser
from time import time
import os
# import argparse

import utils

config=configparser.ConfigParser()
config.read("../config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')

tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(path_to_store_preprocessed_data,"tokenizer"))

# parser = argparse.ArgumentParser()
# parser.add_argument('--chunk_num', type=int, required=True)
# args = parser.parse_args()

# chunk_num = args.chunk_num

for chunk_num in range(7,10):

    s=time()
    with open(os.path.join(path_to_store_preprocessed_data, "filtered_and_truncated", str(chunk_num) + "_texts.json"), "r") as f:
        text_list = json.load(f)
        
    tokenized_list=[]
    
    print(f"Processing chunk_num={chunk_num}...\n")
    for i in range(len(text_list)):
        
        utils.display_progress(i, len(text_list))
        
        text=text_list[i]
        ids_list = tokenizer.encode(text)
        tokenized_list.append(ids_list)
    
    tokenized_folder=os.path.join(path_to_store_preprocessed_data,"tokenized")
    if not os.path.exists(tokenized_folder):
        os.makedirs(tokenized_folder)
    
    with open(os.path.join(tokenized_folder, str(chunk_num) + "_tokenized.json"), "w") as f:
        json.dump(tokenized_list, f) 
        
    e=time()
    print(f"\nrunning_time={round(e-s)}sec\n\n\n")
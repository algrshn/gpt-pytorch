import configparser
import json
from time import time
import sys
import argparse
import os

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_num', type=int, required=True)
args = parser.parse_args()


chunk_num = args.chunk_num

s_time=time()

config=configparser.ConfigParser()
config.read("../config.txt")

path_to_original_data=config.get('preprocessing', 'path_to_original_data')
path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')
pile_sets_to_use=config.get('preprocessing', 'pile_sets_to_use').split("|")
max_len=int(config.get('preprocessing', 'max_len_words'))
min_len=int(config.get('preprocessing', 'min_len_words'))

    
# if(chunk_num>11):
#     chunk_location="data1"
# else:
#     chunk_location="data2"

# with open("/media/alex/" + chunk_location + "/GPT2/ThePile/pile/train/" + str(chunk_num) + ".jsonl", "r") as f:
#     contents = f.readlines()

filename = str(chunk_num) + ".jsonl"    
with open(os.path.join(path_to_original_data,filename), "r") as f:
    contents = f.readlines()
    
set_list=[]
text_list=[]
print(f"Processing chunk #{chunk_num} ...\n")
for i in range(len(contents)):
    
    utils.display_progress(i, len(contents))
    
    e = json.loads(contents[i])

    pile_set_name=e["meta"]["pile_set_name"]
    
    if(pile_set_name in pile_sets_to_use):

        lst=e["text"].split()
        if(len(lst)>=min_len):
            if(len(lst)>max_len):
                truncated_lst=lst[:max_len]
            else:
                truncated_lst=lst
            truncated_text=" ".join(truncated_lst)
        
            text_list.append(truncated_text)
            set_list.append(pile_set_name)

filtered_and_truncated_folder=os.path.join(path_to_store_preprocessed_data,"filtered_and_truncated")
if not os.path.exists(filtered_and_truncated_folder):
    os.makedirs(filtered_and_truncated_folder)

with open(os.path.join(filtered_and_truncated_folder, str(chunk_num) + "_texts.json"), "w") as f:
    json.dump(text_list, f)
                
with open(os.path.join(filtered_and_truncated_folder, str(chunk_num) + "_sets.json"), "w") as f:
    json.dump(set_list, f)                

print("\n\n")            
print(f"len(text_list)={len(text_list)}")
print(f"len(set_list)={len(set_list)}")
print("\n")

e_time=time()

print(f"run_time={round(e_time-s_time)}sec")


import configparser
import json
import argparse
import numpy as np
import pickle
import os
import random

import utils

config=configparser.ConfigParser()
config.read("../config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')
data_from_what_chunk_to_use=int(config.get('val', 'data_from_what_chunk_to_use'))
chunk_num_val=int(config.get('val', 'chunk_num_val'))
num_of_docs=int(config.get('val', 'num_of_docs'))

batched_folder=os.path.join(path_to_store_preprocessed_data,"batched")

with open(os.path.join(batched_folder, str(data_from_what_chunk_to_use) + "_batched.pickle"), "rb") as f:
    data = pickle.load(f)

random.shuffle(data)

output=[]

docs_processed_so_far=0
for i in range(len(data)):
    np_array=data[i]
    output.append(np_array)
    docs_processed_so_far += np_array.shape[0]
    if docs_processed_so_far >= num_of_docs:
        break

with open(os.path.join(batched_folder, str(chunk_num_val) + "_batched.pickle"), "wb") as f:
    pickle.dump(output, f)
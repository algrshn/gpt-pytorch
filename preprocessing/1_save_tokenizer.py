from transformers import GPT2Tokenizer
import configparser
import os

config=configparser.ConfigParser()
config.read("../config.txt")

path_to_store_preprocessed_data=config.get('preprocessing', 'path_to_store_preprocessed_data')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

special_tokens_dict = {}
special_tokens_dict["bos_token"] = "<|bos|>"   
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.save_pretrained(os.path.join(path_to_store_preprocessed_data,"tokenizer"))
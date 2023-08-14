import pandas as pd
import os
from tqdm import tqdm
import math

import torch
from transformers import XLMRobertaTokenizer

# tokenizer load and add token
ckpt = "xlm-roberta-base"

tokenizer = XLMRobertaTokenizer.from_pretrained(ckpt)
print(f"\'{ckpt}\' tokenizer is loaded.")
print(f"XLM-R-base Vocab Size : {tokenizer.vocab_size}")

def tokenize(data_path):

    df = pd.read_csv(data_path, sep=";")
    df = df.head(100) # only for testing!

    tokenized_values = []
    for i, ans in tqdm(enumerate(df.Value)):
        if type(ans) != str:
            if math.isnan(ans):
                ans = ""
        result_tokens = tokenizer.tokenize(ans)
        tokenized_values.append(result_tokens)

    df['tokenized_value'] = tokenized_values
    return df

if __name__=="__main__":

    # file_path = "ePIRLS16_data_v1_dipf.csv"
    # save_path = "ePIRLS16_data_v1_dipf_tkn.csv"
    file_path = "../051.Data/411.merged_CSV/data.csv"
    save_path = "../051.Data/411.merged_CSV/data_tkn.csv"
    tokenized_df = tokenize(file_path)
    tokenized_df.to_csv(save_path, index=False)

    print(f"Tokenized file is saved as {save_path}!")
import time

import pandas as pd
import numpy as np
from datetime import datetime

import text_similarity
import parallelprozessing


# a very basic variance measure: the percentage of unique answers.
# there are indeed completely repetitive answers, this is not an artifact of large numbers of empty answers
def getFractionUnique(df):
    all = len(df)
    unique = len(df["Value"].unique())
    # print(df["Value"].value_counts()) # for sanity checking
    # print(all, unique)
    return unique/all


def getMeanVal(df, func):
    step_size = 24
    values = [str(val) for val in df["Value"]]

    def run(start):
        total = 0
        t_len = 0
        for si in range(min(step_size, len(values)-start)):
            v0 = values[start+si]
            for vi in values[start+si+1:]:
                total += func(v0, vi)[1]
            t_len += len(values) - 1 - (start + si)
        return total, t_len

    tasks = [[i] for i in range(0, len(values)-1, step_size)]
    res = parallelprozessing.run_mono(run, tasks)
    total_val = sum([r[0] for r in res])
    total_count = sum([r[1] for r in res])
    return total_val / total_count  # mean val


df = pd.read_csv("../051.Data/411.merged_CSV/data.CSV", sep=";")
print(df)
print(df.groupby(["Language"])["Value"].count())
print(df.groupby(["Variable"])["Value"].count())
print(df.groupby(["Language", "Variable"])["Value"].count())

# Tabelle Language vs Variable in each cell is the number of answers
table = pd.pivot_table(df, values='Value', index=['Language'], columns=['Variable'], aggfunc=len)
print(table)
print("\n__________________________")

#exit()

# create new table for variability measures

df_var = pd.DataFrame(columns = ["Language", "Variable", "numAnswers", "meanLen"])
for lang in df['Language'].unique():
    #print(lang)
    for var in df['Variable'].unique():
        #print(var)
        df_prompt = df[(df['Language'] == lang) & (df['Variable'] == var)]
        if len(df_prompt) > 3000:
            continue
        print(lang, var, len(df_prompt))

        len_list = [len(str(val)) for val in df_prompt["Value"]]
        meanLen = sum(len_list)/len(len_list)


        #print("{:.4f}".format(num))
        df_var.loc[len(df_var.index)] = [lang, var, len(df_prompt), meanLen]
        df_var.to_csv('variance.tsv', sep="\t")

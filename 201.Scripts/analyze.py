import os
import time

import pandas as pd
import numpy as np
from datetime import datetime

import text_similarity
import corpusVariance
import parallelprozessing


def getMeanLen(df):
    len_list = [len(str(val)) for val in df["Value"]]
    meanLen = sum(len_list) / len(len_list)
    return meanLen


def getMeanMeasureValues(df, measures):
    measure_names = [name for name in measures.keys()]
    values = [str(val) for val in df["Value"]]
    marker = ['IDSTUD', 'LanguageCode', 'Language', 'CountryIsoCode', 'Variable', 'AssessmentResultId', 'SessionId']
    marker_values = [[val for val in df[mark]] for mark in marker]
    df_res = pd.DataFrame(columns=[m + "_A" for m in marker] + [m + "_B" for m in marker] + [m for m in measures.keys()])
    lang = [l for l in df["Language"]][0]
    prompt = [p for p in df["Variable"]][0]

    total = [0 for _ in measure_names]
    t_len = 0

    for si in range(len(values)):
        v0 = values[si]
        for si2 in range(si+1, len(values)):
            vi = values[si2]
            measure_res = [func(v0, vi)[1] for func in [measures.get(m) for m in measure_names]]
            df_res.loc[len(df_res.index)] = [val[si] for val in marker_values] + [val[si2] for val in marker_values] + measure_res
            for i in range(len(total)):
                total[i] += measure_res[i]
        t_len += len(values) - 1 - si

    try:
        os.makedirs("results/" + lang)
    except FileExistsError:
        pass
    df_res.to_csv("results/"+lang+"/"+prompt+".tsv", sep="\t")

    return [v/t_len for v in total]


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

#TODO: alle Maße auch noch auf String-Ebene
text_based_measures = {
    "GST": text_similarity.gst,
    "LCS": text_similarity.longest_common_substring,
    "Levenstein": text_similarity.levenshtein_distance,
    "VCos": text_similarity.vector_cosine
}

corpus_based_measures = {
    "meanLen": getMeanLen,
    "FractionUnique": corpusVariance.getFractionUnique,
    "TTR": corpusVariance.computeTTR
}


def run_task(lang, var):
    df_prompt = df[(df['Language'] == lang) & (df['Variable'] == var)]
    print(lang, var, len(df_prompt))

    cb_vals = [corpus_based_measures.get(m)(df_prompt) for m in corpus_based_measures.keys()]
    tb_vals = getMeanMeasureValues(df_prompt, text_based_measures)

    return [lang, var, len(df_prompt)] + cb_vals + tb_vals


tasks_args = []

df_var = pd.DataFrame(columns = ["Language", "Variable", "numAnswers"] + [m for m in corpus_based_measures.keys()] + ["mean" + m for m in text_based_measures.keys()])
for lang in df['Language'].unique():
    #print(lang)
    for var in df['Variable'].unique():
        tasks_args.append((lang, var))

results = parallelprozessing.run_mono(run_task, tasks_args)
#print("{:.4f}".format(num))
for res in results:
    df_var.loc[len(df_var.index)] = res
df_var.to_csv('variance.tsv', sep="\t")

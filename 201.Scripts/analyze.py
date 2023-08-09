import os
from csv import writer as csv_writer

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


def getMeanMeasureValues(df, measure, verbose=False):
    values = [str(val) for val in df["Value"]]
    ids = [str(id) for id in df["IDSTUD"]]
    lang = [l for l in df["Language"]][0]
    prompt = [p for p in df["Variable"]][0]

    total = 0
    t_len = 0

    try:
        os.makedirs("results/" + lang + "/" + prompt)
    except FileExistsError:
        pass
    tsv_file = csv_writer(open("results/"+lang+"/"+prompt+"/"+measure+".tsv", "w+"), delimiter="\t")
    tsv_file.writerow([measure] + ids)

    for si in range(len(values)):
        v0 = values[si]
        line = [ids[si]] + ["-" for _ in range(si)]
        for si2 in range(si+1, len(values)):
            vi = values[si2]
            measure_res = text_based_measures.get(measure)(v0, vi)[1]
            line.append(measure_res)
            total += measure_res
        t_len += len(values) - 1 - si
        tsv_file.writerow(line)
        if verbose:
            print("Done", si+1, "/", len(values))

    return total/t_len


df = pd.read_csv("../051.Data/411.merged_CSV/filtered_data.tsv", sep="\t")

print(df)
print(df.groupby(["Language"])["Value"].count())
print(df.groupby(["Variable"])["Value"].count())
print(df.groupby(["Language", "Variable"])["Value"].count())

# Tabelle Language vs Variable in each cell is the number of answers
table = pd.pivot_table(df, values='Value', index=['Language'], columns=['Variable'], aggfunc=len)
print(table)
print("\n__________________________")

#exit()

text_based_measures = {
    "GST": text_similarity.gst,
#    "LCS": text_similarity.longest_common_substring,
#    "Levenstein": text_similarity.levenshtein_distance,
#    "VCos": text_similarity.vector_cosine
}

corpus_based_measures = {
#    "meanLen": getMeanLen,
#    "FractionUnique": corpusVariance.getFractionUnique,
#    "TTR": corpusVariance.computeTTR
}


def run_task(lang, var):
    df_prompt = df[(df['Language'] == lang) & (df['Variable'] == var)]
    print(lang, var, len(df_prompt))

    cb_vals = [corpus_based_measures.get(m)(df_prompt) for m in corpus_based_measures.keys()]
    tb_vals = [getMeanMeasureValues(df_prompt, m) for m in text_based_measures.keys()]

    return [lang, var, len(df_prompt)] + cb_vals + tb_vals


if __name__ == "__main__":
    tasks_args = []

    df_var = pd.DataFrame(columns = ["Language", "Variable", "numAnswers"] + [m for m in corpus_based_measures.keys()] + ["mean" + m for m in text_based_measures.keys()])
    for lang in df['Language'].unique():
        #print(lang)
        for var in df['Variable'].unique():
            tasks_args.append((lang, var))

    results = parallelprozessing.run_parallel(run_task, tasks_args)
    #print("{:.4f}".format(num))
    for res in results:
        df_var.loc[len(df_var.index)] = res
    df_var.to_csv('variance.tsv', sep="\t")

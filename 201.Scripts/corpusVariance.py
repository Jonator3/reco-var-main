import pandas as pd
from nltk.tokenize import word_tokenize
import random
from statistics import mean
from datetime import datetime


# This file contains variance metrics considering not pairwise similarity (between texts in a corpus) but metrics wherethe whole corpus is needed as input.

tokensPerSample = 1000
iterations = 1000


# a very basic variance measure: the percentage of unique answers.
# there are indeed completely repetitive answers, this is not an artifact of large numbers of empty answers
def getFractionUnique(df):
    all = len(df)
    unique = len(df["Value"].unique())
    # print(df["Value"].value_counts()) # for sanity checking
    # print(all, unique)
    return unique/all


def computeIndividualTTR(tokenList):
    return len(set(tokenList))/len(tokenList)


# sample a specific number of tokens multiple times and compute TTR for each subsample, return the average
def computeTTR(df, tokenspersample=tokensPerSample, iter=iterations):
    values = [str(val) for val in df["Value"]]
    all_tokens = []
    for value in values:
        tokens = word_tokenize(value)
        tokens = [x.lower() for x in tokens]
        all_tokens = all_tokens + tokens
    # print(all_tokens)
    ttrs = []
    for i in range(iter):
        sample = random.sample(all_tokens, tokenspersample)
        #print(sample)
        ttr = computeIndividualTTR(sample)
        ttrs.append(ttr)
    # print(ttrs)
    # print(mean(ttrs))
    return mean(ttrs)


if __name__ == "__main__":
    #Testcase
    print(datetime.now())

    df = pd.read_csv("../051.Data/411.merged_CSV/data.CSV", sep=";")
    df_var = pd.DataFrame(columns = ["Language", "Variable", "numAnswers", "FractionUnique", "TTR"])
    for lang in df['Language'].unique():
        print(lang)
        for var in df['Variable'].unique():
            print(var)
            df_prompt = df[(df['Language'] == lang) & (df['Variable'] == var)]
            fractionUnique = getFractionUnique(df_prompt)
            ttr = computeTTR(df_prompt, tokensPerSample, iterations)
            df_var.loc[len(df_var.index)] = [lang, var, len(df_prompt), fractionUnique, ttr]
            # break # quit after the first language-prompt-pair
        break  # quit after the first language
    df_var.to_csv('variance_corpus.tsv', sep="\t")
    print(datetime.now())
import pandas as pd
import csv
import os

df_no_easy = pd.read_csv('data/Complete_dataset_thesis.txt', sep="/n", header=None)
df_no_hard = pd.read_csv('data/Complete_dataset_thesis.txt', sep="/n", header=None)

for i, row in df_no_easy.iterrows():
    if i % 2 == 0:
        df_no_easy = df_no_easy.drop(i)
    else:
        df_no_hard= df_no_hard.drop(i)

if os.path.isfile('data/no_easy_formulation.txt'):
    print("no_easy_formulation.txt already exist.")
else:
    df_no_easy.to_csv('data/no_easy_formulation.txt',index=False,header=False, sep="\n", quoting=csv.QUOTE_NONE)

if os.path.isfile('data/no_hard_formulation.txt'):
    print("no_hard_formulation.txt already exist.")
else:
    df_no_hard.to_csv('data/no_hard_formulation.txt',index=False,header=False, sep="\n", quoting=csv.QUOTE_NONE)

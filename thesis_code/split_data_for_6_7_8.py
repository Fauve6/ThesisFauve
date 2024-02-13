import pandas as pd
import numpy as np
import csv

df = pd.read_csv('data/Complete_dataset_thesis.txt', sep="/n", header=None)
df_easy = pd.DataFrame(columns=[0])
df_medium = pd.DataFrame(columns=[0])
df_hard = pd.DataFrame(columns=[0])
for i in range(0, len(df)):
    if i < 80:
        df_temp = df.iloc[[i]]
        df_easy = pd.concat([df_easy, df_temp], ignore_index=True)
    elif 79 < i < 160:
        df_temp = df.iloc[[i]]
        df_medium = pd.concat([df_medium, df_temp], ignore_index=True)
    elif i >= 159:
        df_temp = df.iloc[[i]]
        df_hard = pd.concat([df_hard, df_temp], ignore_index=True)

train_medium_hard = pd.concat([df_medium, df_hard])
train_easy_medium = pd.concat([df_easy, df_medium])
train_easy_hard = pd.concat([df_easy, df_hard])

train_medium_hard.to_csv('data/train/train_medium_hard_sql.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

train_easy_medium.to_csv('data/train/train_easy_medium_sql.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

train_easy_hard.to_csv('data/train/train_easy_hard_sql.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

df_easy.to_csv('data/test/test_easy_sql.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

df_medium.to_csv('data/test/test_medium_sql.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

df_hard.to_csv('data/test/test_hard_sql.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)


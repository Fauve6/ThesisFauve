import pandas as pd
import numpy as np
import csv

df = pd.read_csv('data/no_hard_formulation.txt', sep="/n", header=None)

level_list = []
for i in range(0, len(df)):
    if i < 40:
        level_list.append("e")
    elif 39 < i < 80:
        level_list.append("m")
    elif i > 79:
        level_list.append("h")

df['level'] = level_list

# Set a seed for reproducibility
np.random.seed(0)

# Find unique levels
unique_levels = df['level'].unique()

# Initialize empty DataFrames for each level
e_df = df[df['level'] == 'e']
m_df = df[df['level'] == 'm']
h_df = df[df['level'] == 'h']

# Calculate the number of rows for each level in the first split (70%)
num_e_first_split = int(0.7 * len(e_df))
num_m_first_split = int(0.7 * len(m_df))
num_h_first_split = int(0.7 * len(h_df))

# Randomly shuffle the rows for each level
e_df = e_df.sample(frac=1, random_state=0).reset_index(drop=True)
m_df = m_df.sample(frac=1, random_state=0).reset_index(drop=True)
h_df = h_df.sample(frac=1, random_state=0).reset_index(drop=True)

# Select rows for the first split
e_first_split = e_df[:num_e_first_split]
m_first_split = m_df[:num_m_first_split]
h_first_split = h_df[:num_h_first_split]

# Concatenate the first split DataFrames for each level
first_split_df = pd.concat([e_first_split, m_first_split, h_first_split])

# Create the second split DataFrames for each level
e_second_split = e_df[num_e_first_split:]
m_second_split = m_df[num_m_first_split:]
h_second_split = h_df[num_h_first_split:]

# Concatenate the second split DataFrames for each level
second_split_df = pd.concat([e_second_split, m_second_split, h_second_split])

# Drop the 'level' column
train = first_split_df.drop(columns=['level'])
test = second_split_df.drop(columns=['level'])

# Write the first split DataFrame to a text file
train.to_csv('data/train/train_no_hard_formulation.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

# Write the second split DataFrame to a text file
test.to_csv('data/test/test_no_hard_formulation.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

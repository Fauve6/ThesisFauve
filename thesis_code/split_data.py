import pandas as pd
import numpy as np
import csv

df = pd.read_csv('data/Complete_dataset_thesis.txt', sep="/n", header=None)

j = 1
group_list = []
level_list = []
for i in range(0, len(df)):
    group_list.append(j)
    if i % 2 != 0:
        j += 1
    if i < 80:
        level_list.append("e")
    elif 79 < i < 160:
        level_list.append("m")
    elif i > 159:
        level_list.append("h")

df['group'] = group_list
df['level'] = level_list

# Set a seed for reproducibility
np.random.seed(0)

# Find unique levels
unique_levels = df['level'].unique()

# Initialize empty DataFrames for each level
e_df = pd.DataFrame()
m_df = pd.DataFrame()
h_df = pd.DataFrame()

# Initialize empty DataFrames for the first split
first_split_df = pd.DataFrame(columns=df.columns)

# Split the data based on levels while keeping groups together
for level in unique_levels:
    level_df = df[df['level'] == level]
    unique_groups = level_df['group'].unique()

    # Randomly shuffle the unique groups
    np.random.shuffle(unique_groups)

    num_groups_first_split = int(0.7 * len(unique_groups))
    groups_first_split = unique_groups[:num_groups_first_split]

    mask_first_split = level_df['group'].isin(groups_first_split)

    e_df = pd.concat([e_df, level_df[mask_first_split]]) if level == 'e' else e_df
    m_df = pd.concat([m_df, level_df[mask_first_split]]) if level == 'm' else m_df
    h_df = pd.concat([h_df, level_df[mask_first_split]]) if level == 'h' else h_df

# Concatenate all the level DataFrames back together
train = pd.concat([e_df, m_df, h_df])

# Create a mask to get the second split DataFrame
test = df.drop(train.index)

# Drop the 'group' column
train = train.drop(columns=['group', 'level'])
test = test.drop(columns=['group', 'level'])

# Write the first split DataFrame to a text file
train.to_csv('data/train_everything.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

# Write the second split DataFrame to a text file
test.to_csv('data/test_everything.jsonl', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

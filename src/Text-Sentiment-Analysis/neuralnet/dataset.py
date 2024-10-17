import subprocess
import os
import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import utils

def download_data():
    # Check if files already exist to skip downloading
    if not os.path.exists('twitter-tweets-sentiment-dataset.zip'):
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'yasserh/twitter-tweets-sentiment-dataset'])
    if not os.path.exists('fifa-world-cup-2022-tweets.zip'):
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'tirendazacademy/fifa-world-cup-2022-tweets'])

    # Unzip the downloaded files
    if os.path.exists('twitter-tweets-sentiment-dataset.zip'):
        subprocess.run(['unzip', '-o', 'twitter-tweets-sentiment-dataset.zip'])
    if os.path.exists('fifa-world-cup-2022-tweets.zip'):
        subprocess.run(['unzip', '-o', 'fifa-world-cup-2022-tweets.zip'])


def load_data(random_state=42):
    download_data()
    df_1 = pd.read_csv("Tweets.csv")
    df_2 = pd.read_csv("fifa_world_cup_2022_tweets.csv")

    df_1 = df_1.dropna()

    # Rename the columns 'Tweet' to 'text' and 'Sentiment' to 'sentiment'
    df_2 = df_2.rename(columns={'Tweet': 'text', 'Sentiment': 'sentiment'})

    # Selecting only the 'text' and 'sentiment' columns from both DataFrames
    df_1_limited = df_1[['text', 'sentiment']]
    df_2_limited = df_2[['text', 'sentiment']]

    # Concatenating the two DataFrames row-wise
    df_combined = pd.concat([df_1_limited, df_2_limited], ignore_index=True)
    df = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def clean(df):
    # Preprocess the text column
    df = df.where((pd.notnull(df)),'')

    # Apply preprocessing function to your text column
    df['cleaned_text'] = df['text'].apply(utils.preprocess_text)

    # Remove rows where 'cleaned_text' is empty or contains only whitespace
    df = df[df['cleaned_text'].str.strip() != '']
    df = df.reset_index(drop=True)
    return df

def split_data(df, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X, y = df['cleaned_text'], df['sentiment'].values
    X = utils.embed(X)
    y = [1 if label == 'neutral' else 0 if label == 'negative' else 2 for label in y]
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    return X_train, y_train, X_test, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size, num_workers):
    train_data = TensorDataset((X_train), (y_train))
    test_data = TensorDataset((X_test), (y_test))

    class_names = ["negative", "neutral", "positive"]

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader, class_names
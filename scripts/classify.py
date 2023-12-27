import json
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import os
import random
import csv
from argparse import ArgumentParser

# Download necessary nltk files
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words.add("rt")

def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "user", raw_text)
    return text

def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)
    return text

def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')
    text = text.replace("https", '')
    return text

def remove_stopwords(raw_text):
    tokenize = nltk.word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = " ".join(text)
    return text

def preprocess_text(raw_text):
    text = change_user(raw_text)
    text = remove_entity(text)
    text = remove_url(text)
    text = remove_noise_symbols(text)
    text = remove_stopwords(text)
    return text

"""
Label logic:
Return 1 
    if tweet (hate_speech + offensive_language > neither) 
Else 0
"""
def preprocess_tweet(row):
    tweet = row['tweet']
    label = 1 if row['hate_speech'] + row['offensive_language'] > row['neither'] else 0

    # call preprocessing function to each tweet
    clean_tweet = preprocess_text(tweet)

    return {'text': clean_tweet, 'label': label}

def main(args):
    data = []
    percent = args.percent

    with open(args.file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        length = sum(1 for _ in reader)
        csvfile.seek(0)
        index = 1

        if args.convert:
            print(str(length) + ' sentences found')

        for row in reader:
            # Apply preprocessing to each row
            processed_row = preprocess_tweet(row)

            data.append(processed_row)

            print(f"Processing file {index}/{length} ------------ ({(index/length)*100:.2f}%)", end="\r")
            index += 1

    random.shuffle(data)

    print("creating JSON's")

    train_data = data[:int(length * (1 - percent / 100))]
    test_data = data[int(length * (1 - percent / 100)):]

    with open(os.path.join(args.save_json_path, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(os.path.join(args.save_json_path, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)

    print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Process CSV file and save data to JSON.")
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to the input CSV file.")
    parser.add_argument("--save_json_path", type=str, required=True,
                        help="Path to the directory for saving JSON files.")
    parser.add_argument("--percent", type=float, default=0.5,
                        help="Percentage for some operation.")
    parser.add_argument("--convert", action="store_true",
                        help="Flag to enable conversion operation.")

    args = parser.parse_args()
    main(args)

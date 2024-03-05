import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer  # Ensure this line is added
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import numpy as np
import re
from tensorflow.keras.utils import to_categorical

# use this if data and labels are together in a csv file
# this assumes first row includes titles and the first col are labels, 2nd col data sequences


def read_sequences_and_labels(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header row
        labels, sequences = [], []
        for row in reader:
            labels.append(int(row[0]))
            sequences.append(row[1])
    return sequences, labels

# use this if data is in it's own txt file comma sep


def read_sequences(filename):
    with open(filename, 'r') as file:
        sequences = file.read().split(',')
    return sequences

# encode proteinc


def encode_sequences(sequences):
    # Create a mapping for the 20 amino acids and special characters
    char_to_int = {char: i for i, char in enumerate('ACDEFGHIKLMNPQRSTVWY', 1)}
    encoded_sequences = [[char_to_int[char]
                          for char in sequence if char in char_to_int] for sequence in sequences]
    return encoded_sequences

# encode dna


def encode_sequences_dna(sequences):
    # Assuming DNA sequences (A, C, G, T), update mapping accordingly
    char_to_int = {char: i for i, char in enumerate('ACGT', 1)}
    encoded_sequences = [[char_to_int[char]
                          for char in sequence if char in char_to_int] for sequence in sequences]
    return encoded_sequences

# be careful of this name


def pad_sequences_p(encoded_sequences, maxlen=None):
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_sequences, maxlen=maxlen, padding='post', truncating='post', value=0)
    return padded_sequences

# use this if labels are in own txt file comma separated


def read_labels(filename):
    with open(filename, 'r') as file:
        # file.read().split(',')]
        labels = [int(label) for label in file.read().splitlines()]
    return labels

# TWEETS Sentiment below****************
# for sentiment analysis with labels & tweets


def read_tweets_and_labels(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        labels, tweets = [], []
        for row in reader:
            # labels.append(int(row[0]))
            labels.append(float(row[0]))
            tweets.append(row[1])

    # two lines below only if labels need to be encoded
    scores = np.array(labels)
    encoded_labels = encode_labels(scores)
    # Convert encoded labels to one-hot encoding
    one_hot_labels = to_categorical(encoded_labels, num_classes=3)
    return tweets, one_hot_labels

# preprocess the tweets


def preprocess_tweet(tweet):
    """
    Preprocess a single tweet:
    - Lowercasing
    - Removing URLs, user mentions, and hashtags
    - Optionally, handle emojis in a specific way
    """
    # Convert text to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove hashtags (you can choose to keep the text after the # by removing the '#' in the regex below)
    tweet = re.sub(r'#', '', tweet)
    # Handle emojis (this step is optional and can be customized)
    # For example, you might want to replace certain emojis with words or remove them
    # This step is left as an exercise for the reader and will depend on your specific needs
    return tweet


def preprocess_tweets(tweets, vocab_size=10000, max_length=120, oov_tok="<OOV>"):
    processed_tweets = [preprocess_tweet(tweet) for tweet in tweets]
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(processed_tweets)
    sequences = tokenizer.texts_to_sequences(processed_tweets)
    padded_sequences = pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences, tokenizer

# deals with hate score being in decimals range -5 to 5


def encode_labels(scores):
    """
    Encodes sentiment scores into ordinal categories.
    """
    categories = np.zeros(scores.shape, dtype=int)
    categories[(scores >= -1) & (scores <= 0.5)] = 1
    categories[scores > 0.5] = 2
    return categories


def split_data(sequences, labels):
    return train_test_split(sequences, labels, test_size=0.3, random_state=42)

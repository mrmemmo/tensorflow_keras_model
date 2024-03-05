from tensorflow.keras.models import load_model
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load the saved tokenizer
'''
where higher = more hateful and lower = less hateful. 
> 0.5 is approximately hate speech, ===2
< -1 is counter or supportive speech, and ===0
-1 to +0.5 is neutral or ambiguous.=== 1
'''


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Function to preprocess new tweets using the loaded tokenizer


def preprocess_new_tweets(new_tweets, tokenizer, max_length=120):
    sequences = tokenizer.texts_to_sequences(new_tweets)
    padded_sequences = pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences


# Load the tokenizer and model
tokenizer = load_tokenizer(
    'path_to_file/tokenizer.pickle')
model = load_model(
    'path_to_file/sentiment_analysis_model')

# Example new tweets to classify
new_tweets = ["Tweet on",
              "Tweet two", "tweet three."]

# Preprocess the tweets
padded_new_tweets = preprocess_new_tweets(new_tweets, tokenizer)

# Make predictions
predictions = model.predict(padded_new_tweets)
predicted_categories = np.argmax(predictions, axis=1)

# Print predictions
for tweet, category in zip(new_tweets, predicted_categories):
    print(f"Tweet: '{tweet}' is predicted as Category: {category}")

# Load the model
# Update this to the path where your model is saved

# Now, you can use `loaded_model` to make predictions, evaluate on new data, etc.

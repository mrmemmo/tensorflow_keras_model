from z_data_preprocessing import preprocess_tweets, read_tweets_and_labels, read_sequences_and_labels, read_sequences, encode_sequences, pad_sequences_p, read_labels, split_data
from z_model_training import build_model, train_model
import numpy as np
import pickle


def main():
  
    # use below for tweets/sentiment analysis
    '''     '''
    csv_filename = 'CSV_PATH/hate_sp.csv'

    tweets, labels = read_tweets_and_labels(csv_filename)

    # Preprocess the tweets; this function now also returns the tokenizer used for encoding
    padded_tweets, tokenizer = preprocess_tweets(tweets)
    # labels = np.array(labels)#these should already be in np array format after read_tweets and labels

    # Right after getting the tokenizer, save it to a file
    # Specify your tokenizer save path
    tokenizer_save_path = '/Users/mmemmo/Desktop/Coding/MyPythonCode/AI2024/NeuralNets/files/tokenizer.pickle'
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train, X_test, y_train, y_test = split_data(padded_tweets, labels)

    # Adjust based on the tokenizer's vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    sequence_length = padded_tweets.shape[1]

    model = build_model(vocab_size, sequence_length)
    # Specify your model save path
    model_save_path = '/Users/mmemmo/Desktop/Coding/MyPythonCode/AI2024/NeuralNets/files/sentiment_analysis_model'
    model, history = train_model(
        model, X_train, y_train, X_test, y_test, model_save_path)

 


if __name__ == '__main__':
    main()

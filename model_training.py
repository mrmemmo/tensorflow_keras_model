from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import Dropout


def build_model(vocab_size, sequence_length):
    model = Sequential([
        # Adjusted embedding size
        Embedding(vocab_size + 1, 16, input_length=sequence_length),
        Dropout(0.3),  # Add dropout with a rate of 0.3
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dropout(0.3),  # Additional dropout can be added between dense layers
        Dense(3, activation='softmax')  # Adjusted for 3 classes
    ])
    ''' 
        #original that didn't yeild good results
        Embedding(vocab_size + 1, 16, input_length=sequence_length),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
        
        model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
        '''

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_test, y_test, model_save_path):
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test), batch_size=32)

    # Saving the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    # Extracting the final accuracy and loss from the training set
    final_train_accuracy = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]

    # Extracting the final accuracy and loss from the validation set
    final_val_accuracy = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f"Final Training Accuracy: {final_train_accuracy*100:.2f}%")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy*100:.2f}%")
    print(f"Final Validation Loss: {final_val_loss:.4f}")

    return model, history

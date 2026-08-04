import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the text data
def load_text(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().lower()

    fallback_text = (
        "hello world! this is a fallback text for the character-level lstm demo. "
        "the model will learn simple patterns from this sample when no external file is available."
    )
    return fallback_text.lower()
def prepare_data(text, seq_length):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts([text])
    total_chars = len(tokenizer.word_index) + 1
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_length, 1):
        seq = text[i:i + seq_length]
        label = text[i + seq_length]
        sequences.append([tokenizer.word_index[c] for c in seq])
        next_chars.append(tokenizer.word_index[label])
    X = np.array(sequences)
    y = np.array(next_chars)
    X = pad_sequences(X, maxlen=seq_length)
    X = np.eye(total_chars)[X]
    y = to_categorical(y, num_classes=total_chars)
    return X, y, tokenizer, total_chars
def build_model(seq_length, total_chars):
    model = Sequential([
        LSTM(128, input_shape=(seq_length, total_chars), return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dense(total_chars, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def train_model(model, X, y, epochs=20, batch_size=128):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
def generate_text(model, tokenizer, total_chars, seed_text, seq_length, num_chars):
    reverse_char_index = {i: char for char, i in tokenizer.word_index.items()}
    generated_text = seed_text
    for _ in range(num_chars):
        token_ids = [tokenizer.word_index.get(char, 0) for char in seed_text[-seq_length:]]
        x_pred = pad_sequences([token_ids], maxlen=seq_length)
        x_pred = np.eye(total_chars)[x_pred]
        pred_probs = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(pred_probs)
        next_char = reverse_char_index.get(next_index, ' ')
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text
def main():
    text = load_text('shakespeare.txt')
    seq_length = 40
    X, y, tokenizer, total_chars = prepare_data(text, seq_length)
    model = build_model(seq_length, total_chars)
    train_model(model, X, y, epochs=20, batch_size=128)
    seed_text = text[:seq_length]
    generated_text = generate_text(model, tokenizer, total_chars, seed_text, seq_length, num_chars=400)
    print(generated_text)
if __name__ == "__main__":
    main()

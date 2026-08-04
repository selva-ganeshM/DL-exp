import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def build_and_compile_model(filter_size, stride, padding):
    model = Sequential([
        Conv2D(32, kernel_size=filter_size, strides=stride, padding=padding, activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=filter_size, strides=stride, padding=padding, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=filter_size, strides=stride, padding=padding, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
filter_sizes = [(3, 3), (5, 5)]
strides = [(1, 1), (2, 2)]
paddings = ['valid', 'same']
for filter_size in filter_sizes:
    for stride in strides:
        for padding in paddings:
            print(f"\nExperimenting with filter_size={filter_size}, stride={stride}, padding={padding}")
            
            model = build_and_compile_model(filter_size, stride, padding)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=1)
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test)
            print(f"Test accuracy: {test_accuracy * 100:.2f}%")

            model.summary()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
def generate_sine_wave(seq_length, num_samples):
    x = np.linspace(0, 4 * np.pi, seq_length * num_samples + seq_length)
    data = np.sin(x)
    sequences = []
    for i in range(num_samples):
        start = i * seq_length
        sequences.append(data[start:start + seq_length])
    return np.array(sequences)
seq_length = 50
num_samples = 1000
num_epochs = 10
batch_size = 32

data = generate_sine_wave(seq_length, num_samples)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
X = data[:, :-1]
y = data[:, -1]
X = X.reshape((num_samples, seq_length - 1, 1))
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=num_epochs, batch_size=batch_size)
predictions = model.predict(X)
# Get min and max for denormalization
data_min = np.min(data)
data_max = np.max(data)
data_range = data_max - data_min

# Denormalize y and predictions
y_denorm = y * data_range + data_min
predictions_denorm = predictions.flatten() * data_range + data_min

print(f'Successfully generated predictions for {len(predictions_denorm)} samples')
print(f'Mean true value: {np.mean(y_denorm):.4f}')
print(f'Mean predicted value: {np.mean(predictions_denorm):.4f}')

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(range(len(y_denorm)), y_denorm, label='True Values', linewidth=2)
plt.plot(range(len(predictions_denorm)), predictions_denorm, label='Predicted Values', linewidth=2, alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('RNN Sine Wave Prediction: True vs Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sine_wave_prediction.png')
print('Graph saved as sine_wave_prediction.png')
plt.show()

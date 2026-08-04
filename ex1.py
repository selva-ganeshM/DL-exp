import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)
try:
    encoder = OneHotEncoder(sparse=False)
except TypeError:
    encoder = OneHotEncoder(sparse_output=False)

y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_size = X_train.shape[1]
hidden_size = 10
output_size = y_train.shape[1]
learning_rate = 0.01
epochs = 1000
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
for epoch in range(epochs):
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    loss = np.mean((a2 - y_train) ** 2)
    d_loss_a2 = 2 * (a2 - y_train) / y_train.shape[0]
    d_a2_z2 = sigmoid_derivative(a2)
    d_z2_W2 = a1
    d_z2_a1 = W2
    d_z2 = d_loss_a2 * d_a2_z2
    d_W2 = np.dot(d_z2_W2.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    d_a1_z1 = sigmoid_derivative(a1)
    d_z1_W1 = X_train
    d_z1 = np.dot(d_z2, d_z2_a1.T) * d_a1_z1
    d_W1 = np.dot(d_z1_W1.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
z1 = np.dot(X_test, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
predictions = np.argmax(a2, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
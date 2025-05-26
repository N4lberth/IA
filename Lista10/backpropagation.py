import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def sigmoid(x, deriv=False):
    if deriv:
        s = sigmoid(x)
        return s * (1 - s)
    return 1 / (1 + np.exp(-x))

def tanh(x, deriv=False):
    if deriv:
        return 1 - np.tanh(x)**2
    return np.tanh(x)

def relu(x, deriv=False):
    if deriv:
        return (x > 0).astype(float)
    return np.maximum(0, x)

activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu
}

class MLP:
    def __init__(self, input_size, hidden_size=5, activation='sigmoid', lr=0.1, bias=True):
        self.activation = activation_functions[activation]
        self.use_bias = bias
        self.lr = lr
        self.input_size = input_size

        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, 1)

        if bias:
            self.b1 = np.zeros((1, hidden_size))
            self.b2 = np.zeros((1, 1))
        else:
            self.b1 = None
            self.b2 = None

    def forward(self, X):
        self.z1 = X @ self.W1 + (self.b1 if self.use_bias else 0)
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + (self.b2 if self.use_bias else 0)
        self.a2 = sigmoid(self.z2)  # saída sempre com sigmoid para problemas binários
        return self.a2

    def backward(self, X, y):
        m = y.shape[0]
        dz2 = self.a2 - y
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m if self.use_bias else 0

        dz1 = dz2 @ self.W2.T * self.activation(self.z1, deriv=True)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m if self.use_bias else 0

        self.W2 -= self.lr * dW2
        self.W1 -= self.lr * dW1
        if self.use_bias:
            self.b2 -= self.lr * db2
            self.b1 -= self.lr * db1

    def train(self, X, y, epochs=10000, verbose=False):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            losses.append(loss)
            self.backward(X, y)

            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch} Loss: {loss:.5f}")
        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

def generate_data(func, n):
    X = np.array(list(product([0, 1], repeat=n)))
    if func == "AND":
        y = np.array([[int(all(x))] for x in X])
    elif func == "OR":
        y = np.array([[int(any(x))] for x in X])
    elif func == "XOR":
        y = np.array([[sum(x) % 2] for x in X])
    else:
        raise ValueError("Função desconhecida.")
    return X, y

def run_experiment():
    func = input("Escolha a função (AND, OR, XOR): ").strip().upper()
    n = int(input("Número de entradas (ex: 2 ou 10): "))
    act = input("Função de ativação (sigmoid, tanh, relu): ").strip().lower()
    lr = float(input("Taxa de aprendizado (ex: 0.01, 0.1): "))
    bias = input("Usar bias? (s/n): ").strip().lower() == 's'

    X, y = generate_data(func, n)
    mlp = MLP(input_size=n, activation=act, lr=lr, bias=bias)
    losses = mlp.train(X, y, verbose=True)

    preds = mlp.predict(X)
    print("\nResultados:")
    for xi, yi, pi in zip(X, y, preds):
        print(f"{xi} -> Esperado: {yi[0]}, Predito: {pi[0]}")

    plt.plot(losses)
    plt.title(f"Função: {func} | Ativação: {act} | LR: {lr} | Bias: {bias}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro quadrático médio")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_experiment()

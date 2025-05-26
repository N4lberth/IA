import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = learning_rate
        self.epochs = epochs
        self.history = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                xi_bias = np.insert(xi, 0, 1)
                output = self.activation(np.dot(self.weights, xi_bias))
                error = yi - output
                self.weights += self.lr * error * xi_bias
                total_error += abs(error)
            self.history.append(self.weights.copy())
            if total_error == 0:
                break

    def plot_decision_boundary(self, X, y, title):
        if len(X[0]) != 2:
            print("Plot disponível apenas para 2 entradas.")
            return

        plt.figure()
        for xi, yi in zip(X, y):
            color = 'blue' if yi == 0 else 'red'
            plt.scatter(xi[0], xi[1], c=color)

        for w in self.history:
            if w[2] != 0:
                x_vals = np.array([0, 1])
                y_vals = -(w[1] * x_vals + w[0]) / w[2]
                plt.plot(x_vals, y_vals, linestyle='--', alpha=0.3)

        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True)
        plt.show()

def generate_data(func_name, n_inputs):
    X = list(product([0, 1], repeat=n_inputs))
    if func_name == "AND":
        y = [int(all(x)) for x in X]
    elif func_name == "OR":
        y = [int(any(x)) for x in X]
    elif func_name == "XOR":
        y = [sum(x) % 2 for x in X]
    else:
        raise ValueError("Função desconhecida.")
    return np.array(X), np.array(y)

def main():
    func_name = input("Escolha a função (AND, OR, XOR): ").strip().upper()
    n = int(input("Número de entradas (ex: 2 ou 10): "))

    X, y = generate_data(func_name, n)

    p = Perceptron(input_size=n)
    p.train(X, y)

    # Resultados
    print("\nPesos finais:", p.weights)
    print("\nTestes:")
    for xi, yi in zip(X, y):
        pred = p.predict(xi)
        print(f"{xi} -> Esperado: {yi}, Predito: {pred}")

    if n == 2:
        p.plot_decision_boundary(X, y, f"Perceptron - {func_name}")

if __name__ == "__main__":
    main()

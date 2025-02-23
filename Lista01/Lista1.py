import numpy as np
import pandas as pd

def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def information_gain(df, attribute, target="Conclusao"):
    values, counts = np.unique(df[attribute], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / counts.sum()) * entropy(df[df[attribute] == value][target])
        for i, value in enumerate(values)
    )
    return entropy(df[target]) - weighted_entropy

data = [
    ["Sim", "Nao", "Nao", "Sim", "Alguns", "RRR", "Nao", "Sim", "Frances", "0-10", "Sim"],
    ["Sim", "Nao", "Nao", "Sim", "Cheio", "R", "Nao", "Nao", "Tailandes", "30-60", "Nao"],
    ["Nao", "Sim", "Nao", "Nao", "Alguns", "R", "Nao", "Nao", "Hamburger", "0-10", "Sim"],
    ["Sim", "Nao", "Sim", "Sim", "Cheio", "R", "Sim", "Nao", "Tailandes", "10-30", "Sim"],
    ["Sim", "Nao", "Sim", "Nao", "Cheio", "RRR", "Nao", "Sim", "Frances", ">60", "Nao"],
    ["Nao", "Sim", "Nao", "Sim", "Alguns", "RR", "Sim", "Sim", "Italiano", "0-10", "Sim"],
    ["Nao", "Sim", "Nao", "Nao", "Nenhum", "R", "Sim", "Nao", "Hamburger", "0-10", "Nao"],
    ["Nao", "Nao", "Nao", "Sim", "Alguns", "RR", "Sim", "Sim", "Tailandes", "0-10", "Sim"],
    ["Nao", "Sim", "Sim", "Nao", "Cheio", "R", "Sim", "Nao", "Hamburger", ">60", "Nao"],
    ["Sim", "Sim", "Sim", "Sim", "Cheio", "RRR", "Nao", "Sim", "Italiano", "10-30", "Nao"],
    ["Nao", "Nao", "Nao", "Nao", "Nenhum", "R", "Nao", "Nao", "Tailandes", "0-10", "Nao"],
    ["Sim", "Sim", "Sim", "Sim", "Cheio", "R", "Nao", "Nao", "Hamburger", "30-60", "Sim"],
]

columns = ["Alternativo", "Bar", "SexSab", "Fome", "Cliente", "Preco", "Chuva", "Res", "Tipo", "Tempo", "Conclusao"]
df = pd.DataFrame(data, columns=columns)

info_gains = {col: information_gain(df, col) for col in df.columns if col != "Conclusao"}

print("Ganho de informação de cada atributo:")
for atributo, ganho in info_gains.items():
    print(f"{atributo}: {ganho:.4f}")

raiz = max(info_gains, key=info_gains.get)
print(f"\nA raiz da árvore é: {raiz}")

subtree_info_gains = {}
for value in df[raiz].unique():
    subset = df[df[raiz] == value]
    gains = {col: information_gain(subset, col) for col in df.columns if col not in ["Conclusao", raiz]}
    if gains:
        subtree_info_gains[value] = max(gains, key=gains.get)

print("\nEstrutura da árvore até o segundo nível:")
for valor, atributo in subtree_info_gains.items():
    print(f"Se {raiz} = {valor}, dividir por {atributo}")

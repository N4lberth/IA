import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Carregar os dados
df = pd.read_csv('train.csv')
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Definir colunas
numeric_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked']

# Preprocessamento
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Transformar os dados
X = preprocessor.fit_transform(X)

# Métodos de balanceamento
smote = SMOTE(random_state=42)
tomek = TomekLinks()
rus = RandomUnderSampler(random_state=42)

# Aplicar balanceamento
X_smote, y_smote = smote.fit_resample(X, y)
X_tomek, y_tomek = tomek.fit_resample(X, y)
X_rus, y_rus = rus.fit_resample(X, y)

# Dividir os dados em treino e teste
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = split_data(X_smote, y_smote)
X_train_tomek, X_test_tomek, y_train_tomek, y_test_tomek = split_data(X_tomek, y_tomek)
X_train_rus, X_test_rus, y_train_rus, y_test_rus = split_data(X_rus, y_rus)

# Treinar e avaliar o modelo
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }

# Resultados
results_smote = train_and_evaluate(X_train_smote, X_test_smote, y_train_smote, y_test_smote)
results_tomek = train_and_evaluate(X_train_tomek, X_test_tomek, y_train_tomek, y_test_tomek)
results_rus = train_and_evaluate(X_train_rus, X_test_rus, y_train_rus, y_test_rus)

print("SMOTE Results:", results_smote)
print("TomekLinks Results:", results_tomek)
print("RandomUnderSampler Results:", results_rus)

# Aplicar DSTO-GAN
from dstogan import DSTOGAN
dstogan = DSTOGAN()
X_np = X.toarray() if hasattr(X, "toarray") else X
y_np = y.to_numpy()
X_dsto, y_dsto = dstogan.fit_resample(X_np, y_np)
X_train_dsto, X_test_dsto, y_train_dsto, y_test_dsto = split_data(X_dsto, y_dsto)
results_dsto = train_and_evaluate(X_train_dsto, X_test_dsto, y_train_dsto, y_test_dsto)
print("DSTO-GAN Results:", results_dsto)

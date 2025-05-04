import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler

# Carregar os dados
train_data = pd.read_csv('train.csv')

# Remover colunas irrelevantes
train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Separar features e target
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Definir as colunas numéricas e categóricas
numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Embarked']

# Pré-processamento para colunas numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())
])

# Pré-processamento para colunas categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combinar os pré-processadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Aplicar o pré-processamento
X_preprocessed = preprocessor.fit_transform(X)

# Converter para DataFrame
X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())

# Métodos de balanceamento
smote = SMOTE(random_state=42)
tomek = TomekLinks()
rus = RandomUnderSampler(random_state=42)

# Aplicar balanceamento
X_smote, y_smote = smote.fit_resample(X_preprocessed, y)
X_tomek, y_tomek = tomek.fit_resample(X_preprocessed, y)
X_rus, y_rus = rus.fit_resample(X_preprocessed, y)

# Função para dividir os dados em treino e teste
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = split_data(X_smote, y_smote)
X_train_tomek, X_test_tomek, y_train_tomek, y_test_tomek = split_data(X_tomek, y_tomek)
X_train_rus, X_test_rus, y_train_rus, y_test_rus = split_data(X_rus, y_rus)

# Função para aplicar imputação
def apply_imputation(X_train, X_test, imputer):
    if X_train.isnull().values.any():  # Verifica se há valores ausentes
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        return X_train_imputed, X_test_imputed
    return X_train, X_test  # Retorna os dados originais se não houver valores ausentes

# Função para treinar e avaliar o modelo
def train_and_evaluate(X_train, X_test, y_train, y_test, method_name):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{method_name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))

# Avaliação com diferentes estratégias de balanceamento
train_and_evaluate(X_train_smote, X_test_smote, y_train_smote, y_test_smote, "SMOTE")
train_and_evaluate(X_train_tomek, X_test_tomek, y_train_tomek, y_test_tomek, "TomekLinks")
train_and_evaluate(X_train_rus, X_test_rus, y_train_rus, y_test_rus, "RandomUnderSampler")

# Aplicação dos métodos de imputação em X_train_smote (como exemplo)
# 1. KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn, X_test_knn = apply_imputation(X_train_smote, X_test_smote, knn_imputer)
train_and_evaluate(X_train_knn, X_test_knn, y_train_smote, y_test_smote, "KNNImputer")

# 2. IterativeImputer
iterative_imputer = IterativeImputer(random_state=42)
X_train_iter, X_test_iter = apply_imputation(X_train_smote, X_test_smote, iterative_imputer)
train_and_evaluate(X_train_iter, X_test_iter, y_train_smote, y_test_smote, "IterativeImputer")

# 3. Extra Trees Imputer (Alternative to MissForest)
from sklearn.ensemble import ExtraTreesRegressor
extra_trees_imputer = IterativeImputer(estimator=ExtraTreesRegressor(), random_state=42)
X_train_et, X_test_et = apply_imputation(X_train_smote, X_test_smote, extra_trees_imputer)
train_and_evaluate(X_train_et, X_test_et, y_train_smote, y_test_smote, "ExtraTreesImputer")

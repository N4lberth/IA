import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregar os dados
df = pd.read_csv("train.csv")

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Modelos
rf_param_space = {
    'n_estimators': (10, 200),
    'max_depth': (1, 50),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20),
    'max_features': [None, 'sqrt', 'log2']
}

dt_param_space = {
    'max_depth': (1, 50),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20),
    'max_features': [None, 'sqrt', 'log2']
}

rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

rf_search = BayesSearchCV(rf_model, rf_param_space, n_iter=50, cv=5, n_jobs=-1, random_state=42)
dt_search = BayesSearchCV(dt_model, dt_param_space, n_iter=50, cv=5, n_jobs=-1, random_state=42)

rf_search.fit(X_train, y_train)
dt_search.fit(X_train, y_train)

y_pred_rf = rf_search.predict(X_test)
y_pred_dt = dt_search.predict(X_test)

def evaluate_model(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred)
    }

rf_metrics = evaluate_model(y_test, y_pred_rf)
dt_metrics = evaluate_model(y_test, y_pred_dt)

# Importância dos atributos
rf_importances = rf_search.best_estimator_.feature_importances_
dt_importances = dt_search.best_estimator_.feature_importances_

# Nomes das features
ohe_categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(ohe_categories)

rf_feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances}).sort_values(by='Importance', ascending=False)
dt_feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': dt_importances}).sort_values(by='Importance', ascending=False)

# Exibir resultados
print("Random Forest Metrics:", rf_metrics)
print("Decision Tree Metrics:", dt_metrics)
print("\nRandom Forest Feature Importances:\n", rf_feature_importance)
print("\nDecision Tree Feature Importances:\n", dt_feature_importance)
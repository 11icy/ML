import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, RocCurveDisplay

# 1. Charger le dataset
df = pd.read_csv("ai4i2020.csv")

# 2. Afficher les infos
print(df.head())
print(df.info())
print(df.columns)

# 3. Prétraitement
df = df.drop(["UDI", "Product ID"], axis=1)
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

X = df.drop(["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
y = df["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP ReLU": MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=300, random_state=42),
    "MLP Tanh": MLPClassifier(hidden_layer_sizes=(64,), activation="tanh", max_iter=300, random_state=42),
    "MLP Sigmoid": MLPClassifier(hidden_layer_sizes=(64,), activation="logistic", max_iter=300, random_state=42)
}

# 6. Entraînement et évaluation
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Modèle": name,
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
display(results_df)

# 7. Graphique comparatif
results_df.plot(x="Modèle", y=["Recall", "F1-score", "AUC-ROC"], kind="bar")
plt.title("Comparaison des performances")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

# 8. Courbes de Loss des MLP
plt.plot(models["MLP ReLU"].loss_curve_, label="ReLU")
plt.plot(models["MLP Tanh"].loss_curve_, label="Tanh")
plt.plot(models["MLP Sigmoid"].loss_curve_, label="Sigmoid")
plt.title("Courbes de Loss des MLP")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 9. Courbes ROC
plt.figure(figsize=(8, 6))

for name, model in models.items():
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=plt.gca()
    )

plt.title("Comparaison des courbes ROC")
plt.show()

# 10. Étude de robustesse avec bruit
X_test_noise = X_test + np.random.normal(
    loc=0,
    scale=0.2,
    size=X_test.shape
)

robust_results = []

for name, model in models.items():
    y_pred = model.predict(X_test_noise)
    y_prob = model.predict_proba(X_test_noise)[:, 1]

    robust_results.append({
        "Modèle": name,
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    })

robust_df = pd.DataFrame(robust_results)

print("Résultats normaux")
display(results_df)

print("Résultats avec bruit")
display(robust_df)

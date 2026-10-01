
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


def execute_pipeline():
    print("[MLOps Pipeline] Démarrage du pipeline...")

    # 1. Charger les données
    raw_data = load_iris(as_frame=True)
    df = raw_data.frame

    # 2. Séparer les variables X et la cible y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 3. Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 4. Entraîner le modèle
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    classifier.fit(X_train, y_train)

    # 5. Évaluer le modèle
    score = classifier.score(X_test, y_test)
    print(
        f"[MLOps Pipeline] Modèle entraîné avec succès ! "
        f"Score : {score:.4f}"
    )

    # 6. Enregistrer le modèle
    os.makedirs("models", exist_ok=True)

    with open("models/iris_model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    print("[MLOps Pipeline] Modèle enregistré dans models/iris_model.pkl")


if __name__ == "__main__":
    execute_pipeline()
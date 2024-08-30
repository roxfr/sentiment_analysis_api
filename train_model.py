import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Charger les données
print("Chargement des données...")
try:
    data = pd.read_csv('data/sentiment140.csv', encoding='latin1', usecols=[0, 5], names=['sentiment', 'text'])
    print(f"Nombre de lignes dans les données: {len(data)}")
except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    raise

# Prétraitement
print("Prétraitement des données...")
data['sentiment'] = data['sentiment'].map({0: 'negative', 4: 'positive'})
X = data['text']
y = data['sentiment']

# Vectorisation des données
print("Vectorisation des données...")
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
print(f"Taille de X_vectorized : {X_vectorized.shape}")

# Diviser les données en ensembles d'entraînement et de test
print("Division des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]}")
print(f"Taille de l'ensemble de test : {X_test.shape[0]}")

# Entraîner le modèle
print("Entraînement du modèle...")
model = LogisticRegression(max_iter=2000) 
try:
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès")
except Exception as e:
    print(f"Erreur lors de l'entraînement du modèle : {e}")
    raise

# Évaluer le modèle
print("Évaluation du modèle...")
from sklearn.metrics import classification_report
try:
    y_pred = model.predict(X_test)
    print("Classification Report :")
    print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"Erreur lors de l'évaluation du modèle : {e}")
    raise

# Sauvegarder le modèle et le vectoriseur
print("Sauvegarde du modèle et du vectoriseur...")
try:
    joblib.dump(model, 'data/trained_models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'data/trained_models/vectorizer.pkl')
    print("Modèle et vectoriseur sauvegardés avec succès !")
except Exception as e:
    print(f"Erreur lors de la sauvegarde du modèle ou du vectoriseur : {e}")
    raise
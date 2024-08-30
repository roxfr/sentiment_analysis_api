import joblib
from tensorflow.keras.models import load_model as load_lstm_model
import pickle

# Chargement du modèle ML
def load_ml_model():
    model = joblib.load('data/trained_models/sentiment_model.pkl')
    vectorizer = joblib.load('data/trained_models/vectorizer.pkl')
    return model, vectorizer

# Chargement du modèle LSTM
def load_lstm_model():
    model = load_lstm_model('data/trained_models/sentiment_lstm_model.h5')
    with open('data/trained_models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

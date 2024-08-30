import joblib

def load_model():
    model = joblib.load('data/trained_models/sentiment_model.pkl')
    vectorizer = joblib.load('data/trained_models/vectorizer.pkl')
    return model, vectorizer

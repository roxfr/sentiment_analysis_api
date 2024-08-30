from sentiment_model import load_model

model, vectorizer = load_model()

def predict_text(text: str):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

def predict_file(file):
    contents = file.file.read().decode('utf-8').splitlines()
    texts_vectorized = vectorizer.transform(contents)
    predictions = model.predict(texts_vectorized)
    results = [{"tweet": tweet, "sentiment": sentiment} for tweet, sentiment in zip(contents, predictions)]
    return results

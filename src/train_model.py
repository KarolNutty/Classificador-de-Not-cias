import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv("data/noticias.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["texto"])
y = data["categoria"]

model = MultinomialNB()
model.fit(X, y)

joblib.dump(model, "modelo_noticias.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Modelo treinado e salvo!")

import joblib

modelo = joblib.load("modelo_noticias.pkl")
vectorizer = joblib.load("vectorizer.pkl")

texto = ["O novo robô da Tesla foi apresentado"]
X = vectorizer.transform(texto)
predicao = modelo.predict(X)

print(f"Categoria prevista: {predicao[0]}")

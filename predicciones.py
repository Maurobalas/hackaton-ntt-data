import pandas as pd
import pickle

# Cargar modelo previamente entrenado
def load_model():
    with open("modelo_income.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Preprocesamiento de datos
def preprocess_data(df):
    # Transformar variables categóricas en numéricas con One-Hot Encoding
    df = pd.get_dummies(df, columns=["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"], drop_first=True)
    
    # Transformar age y hours.per.week en categorías
    age_bins = [0, 25, 40, 60, 100]
    age_labels = ["Joven", "Adulto Joven", "Adulto Medio", "Mayor"]
    df["age_category"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)
    
    hours_bins = [0, 20, 40, 60, 100]
    hours_labels = ["Tiempo Parcial", "Jornada Normal", "Horas Extras", "Trabajo Extremo"]
    df["hours_category"] = pd.cut(df["hours.per.week"], bins=hours_bins, labels=hours_labels, right=True)
    
    return df

# Función para hacer las predicciones
def make_predictions(df):
    # Preprocesar los datos
    df_processed = preprocess_data(df)
    
    # Cargar el modelo y hacer las predicciones
    model = load_model()
    predictions = model.predict(df_processed)
    
    # Agregar las predicciones al DataFrame original
    df["Predicción"] = [">50K" if p == 1 else "≤50K" for p in predictions]
    
    return df
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Cargar el modelo entrenado y el preprocesador guardado
model = joblib.load("modelo_entrenado.pkl")  # Modelo guardado
preprocessor = joblib.load("preprocesador_guardado.pkl")  # Preprocesador guardado

# Función para cargar el archivo de test
def cargar_archivo():
    archivo = st.file_uploader("Sube un archivo CSV de test", type=["csv"])
    if archivo is not None:
        # Leer el archivo CSV
        df_test = pd.read_csv(archivo)
        return df_test
    return None

# Función para preprocesar los datos (aplicar escalado y codificación)
def preprocesar_datos(df_test):
    # Aplicar el preprocesamiento guardado (scaler + onehotencoder)
    X_test = preprocessor.transform(df_test)
    return X_test

# Función para predecir utilizando el modelo cargado
def predecir(modelo, X_test):
    predicciones = modelo.predict(X_test)
    return predicciones

# Título de la aplicación
st.title("Modelo de Clasificación de Ingreso (Streamlit)")

# Instrucciones
st.write("""
    Esta aplicación permite predecir el ingreso (si es superior o no a 50K) 
    utilizando el modelo de clasificación entrenado.
""")

# Cargar el archivo de test
df_test = cargar_archivo()

if df_test is not None:
    st.write("Datos de Test:")
    st.write(df_test.head())  # Mostrar las primeras filas del dataset de test

    # Preprocesar los datos de test
    X_test_preprocesado = preprocesar_datos(df_test)

    # Predecir con el modelo
    predicciones = predecir(model, X_test_preprocesado)

    # Mostrar las predicciones
    df_predicciones = pd.DataFrame(predicciones, columns=["Predicción"])
    st.write("Predicciones:")
    st.write(df_predicciones)

    # Mostrar predicciones como categorías
    df_predicciones["Predicción"] = df_predicciones["Predicción"].map({0: "<=50K", 1: ">50K"})
    st.write("Predicciones como categorías (<=50K o >50K):")
    st.write(df_predicciones)
    
    # (Opcional) Puedes exportar las predicciones a un archivo CSV
    st.download_button(
        label="Descargar predicciones",
        data=df_predicciones.to_csv(index=False),
        file_name="predicciones.csv",
        mime="text/csv"
    )
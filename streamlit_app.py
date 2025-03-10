import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Función para cargar el modelo y el preprocesador
def load_model_and_preprocessor():
    # Cargar el modelo entrenado
    model = joblib.load('modelo_entrenado.pkl')  # Reemplaza con la ruta del modelo guardado
    # Cargar el preprocesador
    preprocessor = joblib.load('scaler_and_preprocessor.pkl')  # Reemplaza con la ruta del preprocesador guardado
    return model, preprocessor

# Función para limpiar el dataset (reemplazar '?' por NaN)
def clean_data(df):
    # Reemplazar '?' por NaN
    df.replace('?', np.nan, inplace=True)
    return df

# Función para preprocesar los datos
def preprocess_data(df, preprocessor):
    # Asegurarse de que las columnas de test coincidan con las del conjunto de entrenamiento
    df = df[X_train.columns]  # Asegúrate de tener las mismas columnas que X_train

    # Aplicar el preprocesamiento al conjunto de datos
    X_processed = preprocessor.transform(df)
    return X_processed

# Función para hacer predicciones
def predict(model, preprocessed_data):
    predictions = model.predict(preprocessed_data)
    return predictions

# Configurar la aplicación de Streamlit
st.title("Predicción de Ingresos")

# Subir archivo CSV de test
uploaded_file = st.file_uploader("Cargar archivo de datos de test", type=["csv"])
if uploaded_file is not None:
    # Leer los datos
    df_test = pd.read_csv(uploaded_file)
    
    # Limpiar los datos
    df_cleaned = clean_data(df_test)
    
    # Cargar el modelo y el preprocesador
    model, preprocessor = load_model_and_preprocessor()
    
    # Preprocesar los datos de test
    X_test_processed = preprocess_data(df_cleaned, preprocessor)
    
    # Hacer las predicciones
    y_pred = predict(model, X_test_processed)
    
    # Mostrar las predicciones
    st.write("Predicciones de ingresos:")
    st.write(y_pred)
    
    # Si quieres mostrar la predicción en formato de clasificación
    y_pred_class = ['>50K' if pred == 1 else '<=50K' for pred in y_pred]
    st.write("Predicción de clase:")
    st.write(y_pred_class)

    # Mostrar algunas estadísticas
    st.write("Estadísticas del Dataset de Test")
    st.write(df_cleaned.describe())
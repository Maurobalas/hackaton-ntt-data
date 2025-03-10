import streamlit as st
import pandas as pd
from eda import make_predictions
from eda import eda

# Título de la aplicación
st.title("Predicción de Ingresos (>50K o ≤50K)")

# Cargar archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV con los datos", type=["csv"])
if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file)
    
    # Mostrar el análisis exploratorio de datos (EDA)
    st.header("Análisis Exploratorio de Datos (EDA)")
    eda(df)
    
    # Hacer predicciones
    st.header("Predicciones de Ingresos")
    df_with_predictions = make_predictions(df)
    
    # Mostrar los resultados de las predicciones
    st.write("Resultados de la predicción:")
    st.write(df_with_predictions[["age", "workclass", "education", "hours.per.week", "Predicción"]])
    
    # Botón para descargar los resultados en formato CSV
    st.download_button(label="Descargar Predicciones", data=df_with_predictions.to_csv(index=False), file_name="predicciones.csv", mime="text/csv")
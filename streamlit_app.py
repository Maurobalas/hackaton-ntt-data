import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ===========================
# Cargar modelo y preprocesador
# ===========================

MODEL_PATH = "/Users/mauro/Documents/MIA/HACKATON NTT DATA/hackaton-ntt-data/modelo_entrenado.pkl"  # Ruta del modelo guardado
PREPROCESSOR_PATH = "/Users/mauro/Documents/MIA/HACKATON NTT DATA/hackaton-ntt-data/scaler_and_preprocessor.pkl"  # Ruta del preprocesador

# Intentar cargar el modelo y el preprocesador
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    st.error(f"Error al cargar el modelo o preprocesador: {e}")
    st.stop()

# ===========================
# Función para cargar el archivo CSV
# ===========================

def cargar_archivo():
    archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if archivo is not None:
        df_test = pd.read_csv(archivo)  # Leer el archivo cargado
        return df_test
    return None

# ===========================
# Función para preprocesar los datos
# ===========================

def preprocesar_datos(df_test):
    try:
        X_test = preprocessor.transform(df_test)
        return X_test
    except Exception as e:
        st.error(f"Error en el preprocesamiento: {e}")
        return None

# ===========================
# Función para realizar la predicción
# ===========================

def predecir(modelo, X_test):
    try:
        predicciones = modelo.predict(X_test)
        return predicciones
    except Exception as e:
        st.error(f"Error al realizar predicciones: {e}")
        return None

# ===========================
# Título y descripción de la aplicación
# ===========================

st.title("Modelo de Clasificación de Ingreso (>50K o ≤50K)")

st.write("""
    Esta aplicación permite predecir si una persona tiene un ingreso superior a 50K 
    usando un modelo de Machine Learning preentrenado.
""")

# ===========================
# Cargar y visualizar datos
# ===========================

df_test = cargar_archivo()

if df_test is not None:
    st.write("### Vista previa del dataset cargado:")
    st.write(df_test.head())  

    # Preprocesar los datos cargados
    X_test_preprocesado = preprocesar_datos(df_test)

    if X_test_preprocesado is not None:
        # Realizar predicción
        predicciones = predecir(model, X_test_preprocesado)

        if predicciones is not None:
            # Convertir predicciones a categorías
            df_test["Predicción"] = ["≤50K" if p == 0 else ">50K" for p in predicciones]

            # Mostrar predicciones
            st.write("### Resultados de las predicciones:")
            st.write(df_test[["age", "workclass", "education", "hours.per.week", "Predicción"]])

            # Botón para descargar resultados
            st.download_button(
                label="Descargar Predicciones",
                data=df_test.to_csv(index=False),
                file_name="predicciones.csv",
                mime="text/csv"
            )

# ===========================
# Exploratory Data Analysis (EDA)
# ===========================

st.title("Exploración de Datos (EDA)")

st.write("""
    Carga un dataset en formato CSV para visualizar estadísticas, correlaciones 
    y distribuciones de las variables.
""")

# ===========================
# Cargar datos para análisis exploratorio
# ===========================

df = cargar_archivo()

if df is not None:
    st.write("### Vista previa de los datos cargados:")
    st.write(df.head())

    # Mostrar estadísticas descriptivas
    st.write("### Estadísticas Descriptivas:")
    st.write(df.describe())

    # ===========================
    # Distribución de Variables Numéricas
    # ===========================

    st.write("### Distribución de Variables Numéricas:")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in num_cols:
        fig = plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribución de {col}")
        st.pyplot(fig)

    # ===========================
    # Matriz de Correlación
    # ===========================

    st.write("### Matriz de Correlación:")
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(fig)

    # ===========================
    # Distribución de Variables Categóricas
    # ===========================

    st.write("### Distribución de Variables Categóricas:")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=45)
        plt.title(f"Distribución de {col}")
        st.pyplot(fig)

    # ===========================
    # Análisis Interactivo con Plotly
    # ===========================

    st.write("### Análisis Interactivo con Plotly:")
    
    # Seleccionar variables para el análisis
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if num_cols and cat_cols:
        col_num = st.selectbox("Selecciona una variable numérica:", num_cols)
        col_cat = st.selectbox("Selecciona una variable categórica:", cat_cols)

        # Crear gráfico de cajas interactivo
        fig = px.box(df, x=col_cat, y=col_num, title=f"Boxplot de {col_num} por {col_cat}")
        st.plotly_chart(fig)
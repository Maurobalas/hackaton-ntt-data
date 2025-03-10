import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Cargar el modelo entrenado y el preprocesador guardado
model = joblib.load("modelo_entrenado.pkl")  # Modelo guardado
preprocessor = joblib.load("scaler_and_preprocessor.pkl")  # Preprocesador guardado

# Función para cargar el archivo de test
def cargar_archivo():
    archivo = st.file_uploader("Sube un archivo CSV de test", type=["csv"])
    if archivo is not None:
        # Leer el archivo CSV
        df_test = pd.read_csv('/Users/mauro/Documents/MIA/HACKATON NTT DATA/hackaton-ntt-data/dataset.csv')
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

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Título de la aplicación
st.title("Exploratory Data Analysis (EDA) en Streamlit")

# Subtítulo
st.write("""
    Esta aplicación permite explorar los datos cargados para un análisis exploratorio.
    Puedes cargar tus datos en formato CSV para visualizarlos, analizar sus características
    y ver su distribución y relaciones.
""")

# Cargar archivo CSV
def cargar_archivo():
    archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if archivo is not None:
        # Leer el archivo CSV
        df = pd.read_csv('/Users/mauro/Documents/MIA/HACKATON NTT DATA/hackaton-ntt-data/test(in).csv')
        return df
    return None

# Función para mostrar estadísticas descriptivas
def mostrar_estadisticas(df):
    st.write("### Estadísticas descriptivas")
    st.write(df.describe())

# Función para mostrar las primeras filas de los datos
def mostrar_datos(df):
    st.write("### Primeras filas del dataset")
    st.write(df.head())

# Función para visualizar la distribución de las variables numéricas
def visualizar_distribuciones(df):
    st.write("### Distribución de las variables numéricas")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in num_cols:
        st.write(f"#### Distribución de {col}")
        fig = plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        st.pyplot(fig)

# Función para ver la correlación de las variables numéricas
def visualizar_correlacion(df):
    st.write("### Matriz de Correlación")
    corr = df.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(fig)

# Función para visualizar las variables categóricas
def visualizar_categoricas(df):
    st.write("### Distribución de las variables categóricas")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        st.write(f"#### Distribución de {col}")
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col)
        st.pyplot(fig)

# Función para mostrar boxplot de variables numéricas
def visualizar_boxplot(df):
    st.write("### Boxplot de las variables numéricas")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in num_cols:
        st.write(f"#### Boxplot de {col}")
        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=col)
        st.pyplot(fig)

# Función para gráficos de dispersión (scatter plot) de variables numéricas
def visualizar_scatter_plots(df):
    st.write("### Gráficos de dispersión (scatter plot)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            st.write(f"#### Scatter plot entre {num_cols[i]} y {num_cols[j]}")
            fig = plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df[num_cols[i]], y=df[num_cols[j]])
            st.pyplot(fig)

# Función para análisis interactivo de gráficos con Plotly
def visualizar_interactivo(df):
    st.write("### Gráfico interactivo con Plotly")
    # Seleccionar una columna numérica y otra categórica
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Seleccionar columnas para el gráfico
    col_num = st.selectbox("Selecciona una columna numérica:", num_cols)
    col_cat = st.selectbox("Selecciona una columna categórica:", cat_cols)

    # Gráfico de barras interactivo de Plotly
    fig = px.box(df, x=col_cat, y=col_num, title=f"Boxplot de {col_num} por {col_cat}")
    st.plotly_chart(fig)

# Página de carga de datos
df = cargar_archivo()

if df is not None:
    st.write("### Vista previa de los datos cargados")
    st.write(df.head())

    # Mostrar estadísticas descriptivas
    mostrar_estadisticas(df)

    # Visualizar distribuciones numéricas
    visualizar_distribuciones(df)

    # Mostrar la matriz de correlación
    visualizar_correlacion(df)

    # Visualizar distribuciones categóricas
    visualizar_categoricas(df)

    # Mostrar boxplots de variables numéricas
    visualizar_boxplot(df)

    # Mostrar gráficos de dispersión (scatter plots)
    visualizar_scatter_plots(df)

    # Análisis interactivo con Plotly
    visualizar_interactivo(df)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Título principal de la aplicación
st.title("📊 Predicción de Ingresos (>50K o ≤50K)")

# Cargar el modelo y el escalador previamente entrenados
def load_model():
    with open("modelo_income.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def load_scaler():
    with open("scaler_income.pkl", "rb") as file:
        scaler = pickle.load(file)
    return scaler

# Preprocesamiento de datos
def preprocess_data(df):
    # Transformar variables categóricas en numéricas con One-Hot Encoding
    df = pd.get_dummies(df, columns=["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex"], drop_first=True)
    
    # Transformar native.country en variable binaria
    df["native_usa"] = df["native.country"].apply(lambda x: 1 if x == "United-States" else 0)
    df.drop(columns=["native.country"], inplace=True)
    
    # Transformar age y hours.per.week en categorías
    age_bins = [0, 25, 40, 60, 100]
    age_labels = ["Joven", "Adulto Joven", "Adulto Medio", "Mayor"]
    df["age_category"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)
    
    hours_bins = [0, 20, 40, 60, 100]
    hours_labels = ["Tiempo Parcial", "Jornada Normal", "Horas Extras", "Trabajo Extremo"]
    df["hours_category"] = pd.cut(df["hours.per.week"], bins=hours_bins, labels=hours_labels, right=True)
    
    # Devolver el DataFrame procesado sin las categorías
    return df.drop(columns=["age_category", "hours_category"])

# Barra lateral para la navegación entre páginas
page = st.sidebar.radio("Menu", ["Introducción", "EDA", "Predicción del Modelo"])

# Página: Introducción
if page == "Introducción":
    st.header("Introducción")
    st.write("Esta aplicación permite predecir si una persona tiene ingresos mayores o menores a $50K basándose en sus características demográficas y laborales.")

# Página: Análisis Exploratorio de Datos (EDA)
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    uploaded_file = st.file_uploader("Sube un archivo CSV con los datos", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Vista previa del dataset:")
        st.write(df.head())

        # Mostrar estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.write(df.describe())

        # Gráficos de distribución
        st.subheader("Distribución de ingresos por categoría")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df["income"], palette="coolwarm", ax=ax)
        st.pyplot(fig)

        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        numeric_df = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# Página: Predicción del Modelo
elif page == "Predicción del Modelo":
    st.header("Predicción del Modelo")
    uploaded_file = st.file_uploader("Sube un archivo CSV con los datos", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Vista previa del dataset:")
        st.write(df.head())
        
        # Preprocesar los datos
        df_processed = preprocess_data(df)
        
        # Cargar modelo y escalador
        model = load_model()
        scaler = load_scaler()

        # Escalar las características
        df_scaled = scaler.transform(df_processed)
        
        # Hacer predicción
        predictions = model.predict(df_scaled)
        
        # Mostrar resultados
        df["Predicción"] = [">50K" if p == 1 else "≤50K" for p in predictions]
        st.write("Resultados de la predicción:")
        st.write(df[["age", "workclass", "education", "hours.per.week", "Predicción"]])
        
        # Descargar resultados
        st.download_button(label="Descargar Predicciones", data=df.to_csv(index=False), file_name="predicciones.csv", mime="text/csv")

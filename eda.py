import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Función para mostrar estadísticas descriptivas y análisis general
def eda(df):
    # Mostrar estadísticas descriptivas
    st.write("Estadísticas descriptivas del dataset:")
    st.write(df.describe())

    # Verificar valores nulos
    st.write("Valores nulos por columna:")
    st.write(df.isnull().sum())

    # Visualización de distribuciones de variables
    st.write("Distribución de la variable 'age':")
    sns.histplot(df['age'], kde=True)
    st.pyplot()

    st.write("Distribución de la variable 'hours.per.week':")
    sns.histplot(df['hours.per.week'], kde=True)
    st.pyplot()

    # Correlación entre variables numéricas
    st.write("Mapa de calor de correlación:")
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()
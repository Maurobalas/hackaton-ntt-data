# 📊 Predicción de Ingresos (>50K o ≤50K)

Este proyecto utiliza **inteligencia artificial** para predecir si una persona tiene ingresos superiores a **$50K** basándose en sus características demográficas y laborales.

La aplicación está desarrollada en **Streamlit**, proporcionando una interfaz sencilla e interactiva para cargar datos, analizar tendencias y generar predicciones.

---

## 🚀 Características principales

- 📂 **Carga de dataset CSV** para análisis y predicciones.
- 📊 **Exploración de datos (EDA)** con gráficos de distribución y matriz de correlación.
- 🔍 **Predicción automatizada** basada en un modelo de machine learning preentrenado.
- 💾 **Descarga de resultados** con predicciones.

---

## 📂 Estructura del proyecto

```
income_prediction/
├── data/
│   └── dataset.csv              # Datos de entrenamiento y predicción
├── models/
│   └── modelo_income.pkl        # Modelo preentrenado para predicción
├── app/
│   └── app.py                   # Código de la aplicación Streamlit
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Documentación del proyecto
└── notebooks/
    └── analysis.ipynb           # Exploración y entrenamiento del modelo
```

---

## 📊 Dataset

El dataset debe contener las siguientes columnas:

- **age**: Edad de la persona.
- **workclass**: Tipo de empleo.
- **education**: Nivel educativo.
- **marital.status**: Estado civil.
- **occupation**: Ocupación laboral.
- **relationship**: Relación con el jefe de familia.
- **race**: Raza de la persona.
- **sex**: Género.
- **capital.gain**: Ganancia de capital.
- **capital.loss**: Pérdida de capital.
- **hours.per.week**: Horas trabajadas por semana.
- **native.country**: País de origen (convertido en variable binaria `native_usa`).
- **income**: Categoría objetivo (>50K o ≤50K).

### 🔹 **Ejemplo de datos:**

| age | workclass | education | marital.status | occupation | income |
|-----|----------|-----------|---------------|------------|--------|
| 34  | Private  | HS-grad   | Never-married | Tech-support | <=50K |
| 45  | Self-emp | Bachelors | Married-civ-spouse | Exec-managerial | >50K |

---

## 🛠 Tecnologías utilizadas

- **Lenguaje**: Python
- **Framework**: Streamlit
- **Librerías principales**:
  - `pandas` (manejo de datos)
  - `scikit-learn` (modelado predictivo)
  - `matplotlib` y `seaborn` (visualización de datos)
  - `streamlit` (interfaz de usuario)

---

## 📌 Instalación y uso

### **1️⃣ Clonar el repositorio**
```bash
git clone https://github.com/tu-repo/income-prediction.git
cd income-prediction
```

### **2️⃣ Instalar dependencias**
```bash
pip install -r requirements.txt
```

### **3️⃣ Ejecutar la aplicación**
```bash
streamlit run app/app.py
```

Esto abrirá la aplicación en tu navegador, permitiéndote cargar datos y generar predicciones en tiempo real.

---

## 🎯 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar el modelo o la aplicación, por favor abre un **pull request** o crea un **issue** en el repositorio.

🚀 ¡Gracias por visitar el proyecto! 🎯

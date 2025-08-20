# app.py (versión mejorada)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import plotly.express as px

# --- Configuración de la página ---
st.set_page_config(
    page_title="Comparación de Clasificadores ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título y descripción ---
st.title("🚀 Comparación de Modelos de Clasificación Interactiva")
st.markdown("Esta aplicación simula un conjunto de datos y evalúa el rendimiento de tres algoritmos de clasificación.")
st.markdown("---")

# --- Barra Lateral para Parámetros ---
st.sidebar.header("⚙️ Parámetros del Dataset y Modelos")

# Sliders y selectbox para controlar los datos y modelos
n_samples = st.sidebar.slider("Número de Muestras", min_value=100, max_value=1000, value=300, step=50)
n_features = st.sidebar.slider("Número de Características", min_value=3, max_value=10, value=6, step=1)
n_neighbors = st.sidebar.slider("Vecinos para KNN", min_value=1, max_value=20, value=5, step=1)
st.sidebar.markdown("---")


# --- Creación de los datos simulados ---
# Generamos los datos basándonos en los parámetros de la barra lateral.
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=int(n_features * 0.7), # Un 70% de las características son informativas
    n_redundant=int(n_features * 0.2), # Un 20% son redundantes
    n_classes=2,
    random_state=42
)
data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
data['target'] = y

st.header("1. Conjunto de Datos Simulados")
st.info(f"Se ha creado un conjunto de datos con **{n_samples} muestras** y **{n_features} columnas**.")

# Sección de EDA en un expander para mantener la UI limpia
with st.expander("🔎 Análisis Exploratorio de Datos (EDA)", expanded=False):
    st.subheader("Visualización del Dataset")
    st.dataframe(data.head())

    st.subheader("Distribución de la Variable Objetivo")
    # Gráfico de barras para la distribución de 'target'
    target_counts = data['target'].value_counts().reset_index()
    target_counts.columns = ['Clase', 'Conteo']
    st.bar_chart(target_counts.set_index('Clase'))

    st.subheader("Gráfico de Dispersión de Características")
    # Gráfico interactivo para ver la relación entre dos características
    features_list = data.columns[:-1].tolist()
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("Eje X", options=features_list)
    with col2:
        feature_y = st.selectbox("Eje Y", options=features_list, index=1 if len(features_list) > 1 else 0)

    fig = px.scatter(data, x=feature_x, y=feature_y, color='target',
                     title=f'Dispersión de {feature_x} vs {feature_y}')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- División y Entrenamiento ---
st.header("2. Entrenamiento y Evaluación de Modelos")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.info(f"El conjunto de datos se ha dividido en entrenamiento ({len(X_train)}) y prueba ({len(X_test)}).")

# Definición de modelos con parámetros personalizables
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
decision_tree = DecisionTreeClassifier(random_state=42)
naive_bayes = GaussianNB()

models = {
    'KNN': knn,
    'Árbol de Decisión': decision_tree,
    'Clasificador Bayesiano': naive_bayes
}

# Bucle para entrenar y evaluar
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# --- Mostrar Resultados ---
st.header("3. Resultados de Precisión (Accuracy)")
st.write("A continuación, se muestra la precisión de cada modelo en el conjunto de prueba:")

results_df = pd.DataFrame(results.items(), columns=['Modelo', 'Precisión'])
# Formateamos la precisión a porcentaje
results_df['Precisión'] = results_df['Precisión'].apply(lambda x: f"{x:.2%}")
st.table(results_df.style.highlight_max(axis=0))
st.success("¡El modelo con mayor precisión ha sido resaltado!")
st.markdown("---")

# --- Conclusión ---
st.header("Conclusión")
st.write("Ahora puedes experimentar con los parámetros en la barra lateral para ver cómo cambian los resultados. ¡Observa cómo la precisión de los modelos se ve afectada al modificar el tamaño del dataset o el número de vecinos en KNN!")

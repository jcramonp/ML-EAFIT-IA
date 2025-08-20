# app.py (versión mejorada con carga de archivos y visualización del árbol)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import plotly.express as px
import graphviz
import pydotplus
from io import StringIO

# --- Configuración de la página ---
st.set_page_config(
    page_title="Comparación de Clasificadores ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título y descripción ---
st.title("🚀 Comparación de Modelos de Clasificación Interactiva")
st.markdown("Carga tu propio archivo CSV para analizar o usa el conjunto de datos simulado.")
st.markdown("---")

# --- Barra Lateral para Parámetros ---
st.sidebar.header("⚙️ Parámetros del Dataset y Modelos")
st.sidebar.markdown("**Carga tu archivo aquí** 👇")

# Sección para cargar el archivo CSV
uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type=["csv"])
st.sidebar.markdown("---")

# Sliders y selectbox para controlar los datos y modelos
n_samples = st.sidebar.slider("Número de Muestras (Simulado)", min_value=100, max_value=1000, value=300, step=50, disabled=(uploaded_file is not None))
n_features = st.sidebar.slider("Número de Características (Simulado)", min_value=3, max_value=10, value=6, step=1, disabled=(uploaded_file is not None))
n_neighbors = st.sidebar.slider("Vecinos para KNN", min_value=1, max_value=20, value=5, step=1)
st.sidebar.markdown("---")

# --- Lógica de carga o simulación de datos ---
if uploaded_file is not None:
    # Si se sube un archivo, usa esos datos
    try:
        data = pd.read_csv(uploaded_file)
        st.header("1. Conjunto de Datos Cargado")
        st.info(f"Se ha cargado un conjunto de datos con **{data.shape[0]} muestras** y **{data.shape[1]} columnas**.")

        # Selector para que el usuario elija la columna objetivo
        target_column = st.selectbox("Selecciona la columna objetivo", options=data.columns)
        
        # Separar características (X) y objetivo (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Convertir a numpy arrays
        X = X.to_numpy()
        y = y.to_numpy()
        
        # Obtener los nombres de las características y las clases
        feature_names = data.drop(columns=[target_column]).columns
        class_names = [str(c) for c in y.unique()]

    except Exception as e:
        st.error(f"Error al leer el archivo. Asegúrate de que es un archivo CSV válido. Error: {e}")
        st.stop()
else:
    # Si no hay archivo, se genera un dataset simulado
    st.header("1. Conjunto de Datos Simulados")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=2,
        random_state=42
    )
    data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
    data['target'] = y
    st.info(f"Se ha creado un conjunto de datos simulado con **{n_samples} muestras** y **{n_features} columnas**.")

    # Obtener los nombres de las características y las clases
    feature_names = data.columns[:-1]
    class_names = ['Clase 0', 'Clase 1']

# Sección de EDA
with st.expander("🔎 Análisis Exploratorio de Datos (EDA)", expanded=False):
    st.subheader("Visualización del Dataset")
    st.dataframe(data.head())

    st.subheader("Distribución de la Variable Objetivo")
    # Gráfico de barras para la distribución de 'target'
    target_counts = data.iloc[:, -1].value_counts().reset_index()
    target_counts.columns = ['Clase', 'Conteo']
    st.bar_chart(target_counts.set_index('Clase'))

    st.subheader("Gráfico de Dispersión de Características")
    # Gráfico interactivo para ver la relación entre dos características
    features_list = data.columns.tolist()
    if uploaded_file is not None:
        features_list.remove(target_column)

    if len(features_list) > 1:
        col1, col2 = st.columns(2)
        with col1:
            feature_x = st.selectbox("Eje X", options=features_list)
        with col2:
            feature_y = st.selectbox("Eje Y", options=features_list, index=1)

        fig = px.scatter(data, x=feature_x, y=feature_y, color=data.columns[-1],
                         title=f'Dispersión de {feature_x} vs {feature_y}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Se necesitan al menos 2 columnas de características para generar un gráfico de dispersión.")

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
results_df['Precisión'] = results_df['Precisión'].apply(lambda x: f"{x:.2%}")
st.table(results_df.style.highlight_max(axis=0))
st.success("¡El modelo con mayor precisión ha sido resaltado!")
st.markdown("---")

# --- Visualización del Árbol de Decisión ---
st.header("4. Visualización del Árbol de Decisión")
st.info("Explora la estructura del Árbol de Decisión y cómo toma sus predicciones.")

dot_data = StringIO()
export_graphviz(
    decision_tree, 
    out_file=dot_data,  
    filled=True, 
    rounded=True,
    special_characters=True,
    feature_names=feature_names,
    class_names=class_names
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
st.graphviz_chart(graph.to_string())

st.markdown("---")

# --- Conclusión ---
st.header("Conclusión")
st.write("Ahora puedes experimentar con los parámetros en la barra lateral o cargar tu propio conjunto de datos para ver cómo se comportan los modelos.")

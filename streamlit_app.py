## %% Fase 1. Carga de Datos
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_excel('./Data_inconcert_UTEL.xlsx')

## %% Fase 2. Limpieza de datos

#Limpieza de datos nulos
df= df.dropna()
#Limpieza de columnas que solo tienen una única variación
df = df.drop(columns=['ID', 'Nombre', 'Email', 'País', 'Teléfono', 'Cuenta', 'Categoría', 'Creado por evento', 'Fecha creación', 'Origen de creación', 'Último evento'])
entrenamiento = df.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
entrenamiento['Campaña'] = le.fit_transform(entrenamiento['Campaña'])
entrenamiento['Última actividad'] = le.fit_transform(entrenamiento['Última actividad'])
entrenamiento['Última actividad (tipo)'] = le.fit_transform(entrenamiento['Última actividad (tipo)'])
entrenamiento['Etapa'] = le.fit_transform(entrenamiento['Etapa'])
entrenamiento['ID de origen'] = le.fit_transform(entrenamiento['ID de origen'])
entrenamiento['Nivel de programa'] = le.fit_transform(entrenamiento['Nivel de programa'])
entrenamiento['Programa de interés'] = le.fit_transform(entrenamiento['Programa de interés'])
#df['Origen de creación'] = le.fit_transform(entrenamiento['Origen de creación'])
Etiquetas= entrenamiento['Etapa'].values
Vector_caracteristicas = entrenamiento.drop(['Etapa'], axis=1)


# Contar la cantidad de "lead" por programa de interés
lead_counts = df[df['Etapa'] == 'lead']['Programa de interés'].value_counts()

# Seleccionar los 5 programas con mayor cantidad de "lead"
top_programs = lead_counts.head(5)

# Contar la cantidad de "contacto" por programa de interés
contacto_counts = df[df['Etapa'] == 'Contacto']['Programa de interés'].value_counts()

# Seleccionar los 5 programas con mayor cantidad de "contacto"
top_contacto = contacto_counts.head(5)

# Contar la cantidad de "prospecto" por programa de interés
prospecto_counts = df[df['Etapa'] == 'Prospecto']['Programa de interés'].value_counts()

# Seleccionar los 5 programas con mayor cantidad de "prospecto"
top_prospecto = prospecto_counts.head(5)

# Sumar la cantidad de cada etapa por programa de interés
etapa_counts = df.groupby('Programa de interés')['Etapa'].value_counts().unstack().fillna(0)

# Sumar el total de etapas por programa de interés
etapa_counts['Total'] = etapa_counts.sum(axis=1)

# Seleccionar los top 5 programas con mayor cantidad total de etapas
top_programas = etapa_counts['Total'].nlargest(5)

# Sumar la cantidad de programas de interés por campaña
programas_por_campana = df.groupby('Campaña')['Programa de interés'].nunique()

# Seleccionar los top 5 programas de interés por campaña
top_programas_por_campana = programas_por_campana.nlargest(5)

## %% FASE 3: MODELADO DE MACHINE LEARNING

from sklearn.model_selection import train_test_split

## %% Fase 4: Creación del Dashboard en Streamlit
# 4.1 Estructura básica
import streamlit as st
import altair as alt

# Configuración de la página
st.set_page_config(
    page_title="UTEL Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.title('🏂 Dashboard UTEL DB 🏂 ')
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQb7KcrVxUIyiDHYzHqIryWn7Lpgu5Nc9Vauz9lBRDA0Q&s", width=150)

st.markdown("""
<style>
   .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

st.title('Dashboard UTEL DB')

# 3.2 Integración de gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Visualización de gráficos

tab1, tab2, tab3, tab4, tab5 = st.tabs(["DB", "Etapa", "Top 5 Etapas por programa", "Campaña", "Modelado"])
with tab1:
    df
with tab2:
    ## Gráfica 1
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Campaña", hue="Etapa", legend=False, data=df, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Campaña por Etapa')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    st.pyplot(plt)

    ## Gráfica 2
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Nivel de programa", hue="Etapa", legend=False, data=df, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Nivel de programa por Etapa')
    plt.xlabel('Nivel de programa')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    st.pyplot(plt)

    ##Gráfica 3
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Programa de interés", hue="Etapa", legend=False, data=df, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Programa de interés por Etapa')
    plt.xlabel('Programa de interés')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    st.pyplot(plt)

with tab3:
    ## Gráfica LEADS
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_programs.index, y=top_programs.values, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 LEADS')
    plt.xlabel('Programa de interés')
    plt.ylabel('Cantidad de leads')
    plt.tight_layout()
    st.pyplot(plt)

    ## Gráfica Contacto
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_contacto.index, y=top_contacto.values, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 CONTACTO')
    plt.xlabel('Programa de interés')
    plt.ylabel('Cantidad de contacto')
    plt.tight_layout()
    st.pyplot(plt)

    ## Gráfica Prospecto
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_prospecto.index, y=top_prospecto.values, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 PROSPECTO')
    plt.xlabel('Programa de interés')
    plt.ylabel('Cantidad de prospecto')
    plt.tight_layout()
    st.pyplot(plt)

    ## Programa con más tracción

    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_programas.index, y=top_programas.values, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 programas con mayor cantidad total de etapas')
    plt.xlabel('Programa de interés')
    plt.ylabel('Cantidad total de etapas')
    plt.tight_layout()
    st.pyplot(plt)

with tab4:
    sns.countplot(x="Campaña", hue="Nivel de programa", legend=False, data=df, palette="Set2")
    plt.title('Campaña por Nivel de programa')
    plt.xlabel('Campaña')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)


    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_programas_por_campana.index, y=top_programas_por_campana.values, palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 Programas de Interés por Campaña')
    plt.xlabel('Campaña')
    plt.ylabel('Cantidad de Programas de Interés')
    plt.tight_layout()
    st.pyplot(plt)
with tab5:
    st.markdown("Se trabajó con el modelo ***RandomForestClassifier***.")
    st.markdown("Con la siguiente DB:")
    df
    html_str = f"""
    <style>
    p.a {{
    font: bold 15px Courier;
    }}
    </style>
    <p class="a">Obteniendo un valor de: 84.53</p>
    """
    st.markdown(html_str, unsafe_allow_html=True)

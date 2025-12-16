"""
SISTEMA INTEGRADO DE ANÁLISIS Y PREDICCIÓN DE PATENTES USPTO
Web App con Streamlit
================================================================
Este sistema integrado:
1. Carga datos desde 66 archivos CSV en Google Cloud Storage
2. Procesa y visualiza datos históricos
3. Implementa Ensemble Learning para predecir patentes 2025-2031
4. Genera visualizaciones y reportes completos
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64
import os

# Verificar si google-cloud-storage está disponible
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    st.warning("⚠ google-cloud-storage no está instalado. Usando datos de ejemplo.")

# Configuración de la página
st.set_page_config(
    page_title="USPTO Patent Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Sistema Integrado de Análisis y Predicción de Patentes USPTO")
st.markdown("---")

# ============================================================================
# 1. CONFIGURACIÓN Y CACHE DE DATOS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Cargando datos desde GCS...")
def cargar_datos_desde_gcs():
    """Carga los 66 archivos CSV desde Google Cloud Storage con cache"""
    
    if not GCS_AVAILABLE:
        st.error("❌ google-cloud-storage no está disponible. Usando datos de ejemplo.")
        return generar_datos_ejemplo()
    
    try:
        # Configurar cliente de GCS
        client = storage.Client(project='warm-physics-474702-q3')
        bucket = client.bucket('patentbucket-maam')
        
        todos_datos = []
        archivos_encontrados = 0
        
        # Cargar los primeros 20 archivos para desarrollo más rápido
        max_archivos = 66  # Reducido para desarrollo más rápido
        
        for i in range(min(66, max_archivos)):
            nombre_archivo = f"{i:012d}.csv"
            
            try:
                blob = bucket.blob(nombre_archivo)
                
                if blob.exists():
                    # Mostrar progreso
                    st.write(f"📂 Cargando archivo [{i:012d}/66]")
                    
                    contenido = blob.download_as_bytes()
                    df_chunk = pd.read_csv(io.BytesIO(contenido))
                    todos_datos.append(df_chunk)
                    archivos_encontrados += 1
                    
                else:
                    st.write(f"⚠ {nombre_archivo} no encontrado")
                    
            except Exception as e:
                st.write(f"❌ Error con {nombre_archivo}: {str(e)[:500]}")
                continue
        
        if not todos_datos:
            st.warning("⚠ No se encontraron archivos en GCS. Usando datos de ejemplo...")
            return generar_datos_ejemplo()
        
        df_completo = pd.concat(todos_datos, ignore_index=True)
        
        # Mensaje de éxito
        st.success(f"✅ Datos cargados exitosamente: {len(df_completo):,} registros de {archivos_encontrados} archivos")
        
        return df_completo
        
    except Exception as e:
        st.error(f"❌ Error de conexión GCS: {str(e)[:100]}")
        st.info("💡 Usando datos de ejemplo para demostración...")
        return generar_datos_ejemplo()

@st.cache_data
def generar_datos_ejemplo():
    """Genera datos de ejemplo realistas para demostración"""
    
    st.info("🔄 Generando datos de ejemplo realistas...")
    
    np.random.seed(42)
    n = 30000  # 30,000 registros de ejemplo para Streamlit (más rápido)
    
    paises = ['US', 'CN', 'JP', 'DE', 'KR', 'GB', 'FR', 'IN', 'CA', 'BR', 
              'TW', 'NL', 'CH', 'SE', 'IT', 'AU', 'MX', 'ES', 'RU', 'SG']
    
    # Distribución realista
    pesos = [0.35, 0.25, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02] + [0.01] * 10
    pesos = pesos[:len(paises)]
    pesos = [p/sum(pesos) for p in pesos]
    
    secciones = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    datos = {
        'assignee_country': np.random.choice(paises, n, p=pesos),
        'section': np.random.choice(secciones, n),
        'patent_date': pd.date_range('2006-01-01', '2021-12-31', periods=n),
        'num_claims': np.random.randint(1, 50, n),
        'classification_level': np.random.choice(['MAIN', 'FURTHER'], n, p=[0.8, 0.2]),
        'ipc_class': [f'{"ABCDEFGH"[np.random.randint(0,8)]}{np.random.randint(10, 99):02d}' for _ in range(n)]
    }
    
    df = pd.DataFrame(datos)
    df['year'] = df['patent_date'].dt.year
    
    st.success(f"📋 Datos de ejemplo generados: {len(df):,} registros (2006-2021)")
    
    return df

@st.cache_data
def preparar_datos_visualizacion(df):
    """Prepara datos para visualizaciones"""
    
    # Asegurar columnas necesarias
    if 'year' not in df.columns and 'patent_date' in df.columns:
        df['patent_date'] = pd.to_datetime(df['patent_date'], errors='coerce')
        df['year'] = df['patent_date'].dt.year
    elif 'year' not in df.columns:
        df['year'] = np.random.randint(2006, 2022, len(df))
    
    # Crear nombres de sección
    seccion_dict = {
        'A': 'Necesidades Humanas',
        'B': 'Operaciones y Transporte', 
        'C': 'Química y Metalurgia',
        'D': 'Textiles y Papel',
        'E': 'Construcción Fija',
        'F': 'Mecánica, Iluminación',
        'G': 'Física',
        'H': 'Electricidad'
    }
    
    if 'section' in df.columns:
        df['section_name'] = df['section'].map(seccion_dict)
    
    # Crear datos agregados por país y año
    df_agregado = df.groupby(['assignee_country', 'year']).agg({
        'assignee_country': 'count',
        'num_claims': 'mean',
        'section': lambda x: x.nunique()
    }).rename(columns={'assignee_country': 'num_patentes'}).reset_index()
    
    return df, df_agregado

# ============================================================================
# 2. FUNCIONES DE VISUALIZACIÓN CON PLOTLY
# ============================================================================

def crear_mapa_mundial_interactivo(df_agregado, year=None, section=None, df_original=None):
    """Crea mapa mundial interactivo con Plotly"""
    
    datos = df_agregado.copy()
    
    # Aplicar filtros
    if year:
        datos = datos[datos['year'] == year]
    
    # Si se filtra por sección, usar datos originales
    if section and df_original is not None:
        datos_seccion = df_original[df_original['section'] == section]
        datos = datos_seccion.groupby(['assignee_country', 'year']).size().reset_index(name='num_patentes')
        if year:
            datos = datos[datos['year'] == year]
    
    # Contar por país
    conteo = datos.groupby('assignee_country')['num_patentes'].sum().reset_index()
    
    # Mapeo de códigos ISO
    codigos_iso = {
        'US': 'USA', 'CN': 'CHN', 'JP': 'JPN', 'DE': 'DEU', 'KR': 'KOR',
        'GB': 'GBR', 'FR': 'FRA', 'IN': 'IND', 'CA': 'CAN', 'BR': 'BRA',
        'MX': 'MEX', 'ES': 'ESP', 'IT': 'ITA', 'RU': 'RUS', 'AU': 'AUS',
        'NL': 'NLD', 'CH': 'CHE', 'SE': 'SWE', 'TW': 'TWN', 'SG': 'SGP'
    }
    
    conteo['iso_a3'] = conteo['assignee_country'].map(codigos_iso)
    
    # Título
    titulo = "🌍 Distribución Global de Patentes USPTO"
    if year:
        titulo += f" - Año {year}"
    if section:
        nombres = {'A': 'Necesidades', 'B': 'Operaciones', 'C': 'Química',
                  'D': 'Textiles', 'E': 'Construcción', 'F': 'Mecánica',
                  'G': 'Física', 'H': 'Electricidad'}
        titulo += f" - Sección {section} ({nombres.get(section, section)})"
    
    # Crear mapa con Plotly
    fig = px.choropleth(
        conteo,
        locations="iso_a3",
        color="num_patentes",
        hover_name="assignee_country",
        hover_data={"num_patentes": True, "iso_a3": False},
        color_continuous_scale="YlOrRd",
        title=titulo,
        labels={'num_patentes': 'Número de Patentes'}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def grafico_tendencia_anual_interactivo(df_agregado, pais=None, seccion=None, df_original=None):
    """Gráfico de evolución anual de patentes con Plotly"""
    
    datos = df_agregado.copy()
    
    # Filtrar por país
    if pais:
        datos = datos[datos['assignee_country'] == pais]
    
    # Filtrar por sección
    if seccion and df_original is not None:
        datos_seccion = df_original[df_original['section'] == seccion]
        datos = datos_seccion.groupby(['year']).size().reset_index(name='num_patentes')
    
    # Agrupar por año si no estamos filtrando por país
    if not pais and not seccion:
        datos = datos.groupby('year')['num_patentes'].sum().reset_index()
    
    # Título
    titulo = "📈 Evolución Anual de Patentes"
    if pais:
        titulo += f" - {pais}"
    if seccion:
        nombres = {'A': 'Necesidades', 'B': 'Operaciones', 'C': 'Química',
                  'D': 'Textiles', 'E': 'Construcción', 'F': 'Mecánica',
                  'G': 'Física', 'H': 'Electricidad'}
        titulo += f" - Sección {seccion} ({nombres.get(seccion, seccion)})"
    
    # Crear gráfico con Plotly
    fig = px.line(
        datos,
        x='year',
        y='num_patentes',
        title=titulo,
        markers=True,
        line_shape='spline'
    )
    
    # Añadir área sombreada
    fig.add_trace(go.Scatter(
        x=datos['year'],
        y=datos['num_patentes'],
        fill='tozeroy',
        fillcolor='rgba(100, 149, 237, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        xaxis_title="Año",
        yaxis_title="Número de Patentes",
        hovermode='x unified',
        height=450
    )
    
    return fig

# ============================================================================
# 3. ENSEMBLE LEARNING PARA STREAMLIT
# ============================================================================

class EnsemblePredictorStreamlit:
    """Clase para predicción usando Ensemble Learning optimizada para Streamlit"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=50,  # Reducido para Streamlit
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=40,  # Reducido para Streamlit
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.feature_columns = None
        self.metrics = {}
        self.ensemble_weights = {'random_forest': 0.6, 'gradient_boosting': 0.4}
    
    def preparar_datos_para_prediccion(self, df_agregado):
        """Prepara datos para entrenamiento de modelos"""
        
        datos = df_agregado.copy()
        
        # Añadir características temporales
        datos['year_squared'] = datos['year'] ** 2
        datos['year_cubed'] = datos['year'] ** 3
        
        # Codificar países
        datos['country_encoded'] = self.encoder.fit_transform(datos['assignee_country'])
        
        # Añadir características por país
        paises = datos['assignee_country'].unique()
        
        for pais in paises:
            datos_pais = datos[datos['assignee_country'] == pais].sort_values('year')
            
            if len(datos_pais) >= 3:
                # Media móvil de 3 años
                datos.loc[datos_pais.index, 'ma_3y'] = datos_pais['num_patentes'].rolling(window=3, min_periods=1).mean()
                
                # Tasa de crecimiento
                datos.loc[datos_pais.index, 'growth_rate'] = datos_pais['num_patentes'].pct_change().fillna(0)
        
        # Rellenar valores faltantes
        datos['ma_3y'] = datos['ma_3y'].fillna(datos['num_patentes'])
        datos['growth_rate'] = datos['growth_rate'].fillna(0)
        
        return datos
    
    def crear_dataset_entrenamiento(self, datos_preparados, horizonte=6):
        """Crea dataset para entrenamiento"""
        
        paises = datos_preparados['assignee_country'].unique()
        muestras = []
        
        for pais in paises:
            datos_pais = datos_preparados[datos_preparados['assignee_country'] == pais].sort_values('year')
            
            if len(datos_pais) >= horizonte + 3:
                for i in range(len(datos_pais) - horizonte):
                    fila_actual = datos_pais.iloc[i]
                    
                    # Características históricas
                    historico = datos_pais.iloc[max(0, i-3):i+1]
                    
                    muestra = {
                        'pais': pais,
                        'year_actual': fila_actual['year'],
                        'country_encoded': fila_actual['country_encoded'],
                        'num_patentes_actual': fila_actual['num_patentes'],
                        'avg_claims': fila_actual['num_claims'],
                        'sections_unique': fila_actual['section'],
                        'mean_3y': historico['num_patentes'].mean() if len(historico) > 0 else 0,
                        'std_3y': historico['num_patentes'].std() if len(historico) > 0 else 0,
                        'growth_3y': historico['growth_rate'].mean() if 'growth_rate' in historico.columns and len(historico) > 0 else 0,
                        'target_6y': datos_pais.iloc[i+6]['num_patentes'] if i+6 < len(datos_pais) else None
                    }
                    
                    muestras.append(muestra)
        
        df_entrenamiento = pd.DataFrame(muestras)
        df_entrenamiento = df_entrenamiento.dropna()
        
        return df_entrenamiento
    
    def entrenar_modelos(self, df_entrenamiento, horizonte=6):
        """Entrena los modelos ensemble"""
        
        # Preparar características y target
        feature_cols = [
            'country_encoded', 'year_actual',
            'num_patentes_actual', 'avg_claims', 'sections_unique',
            'mean_3y', 'std_3y', 'growth_3y'
        ]
        
        available_cols = [col for col in feature_cols if col in df_entrenamiento.columns]
        self.feature_columns = available_cols
        
        X = df_entrenamiento[available_cols]
        y = df_entrenamiento[f'target_{horizonte}y']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar y evaluar cada modelo
        resultados = {}
        
        for nombre, modelo in self.models.items():
            # Validación cruzada simplificada para Streamlit
            cv_scores = cross_val_score(modelo, X_train_scaled, y_train, 
                                       cv=3, scoring='r2', n_jobs=-1)  # Reducido a 3 folds
            
            # Entrenar modelo final
            modelo.fit(X_train_scaled, y_train)
            
            # Evaluar en test
            y_pred = modelo.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.metrics[nombre] = {
                'cv_mean_r2': cv_scores.mean(),
                'cv_std_r2': cv_scores.std(),
                'test_mae': mae,
                'test_rmse': rmse,
                'test_r2': r2
            }
            
            resultados[nombre] = {
                'model': modelo,
                'cv_scores': cv_scores
            }
        
        return X_test, y_test, resultados
    
    def predecir_futuro(self, df_agregado, años_futuros=6, países_interes=None):
        """Genera predicciones para años futuros"""
        
        if países_interes is None:
            top_paises = df_agregado.groupby('assignee_country')['num_patentes'].sum().nlargest(10).index
            países_interes = top_paises.tolist()
        
        predicciones = []
        
        for pais in países_interes:
            datos_pais = df_agregado[df_agregado['assignee_country'] == pais].sort_values('year')
            
            if len(datos_pais) < 3:
                continue
            
            # Último año disponible
            ultimo_año = datos_pais['year'].max()
            ultimos_datos = datos_pais[datos_pais['year'] == ultimo_año].iloc[0]
            
            # Codificar país
            try:
                country_encoded = self.encoder.transform([pais])[0]
            except:
                continue
            
            # Generar predicciones para cada año futuro
            for año_offset in range(1, años_futuros + 1):
                año_futuro = 2021 + año_offset
                
                # Preparar características para predicción
                features = {
                    'country_encoded': country_encoded,
                    'year_actual': año_futuro,
                    'num_patentes_actual': ultimos_datos['num_patentes'],
                    'avg_claims': ultimos_datos['num_claims'],
                    'sections_unique': ultimos_datos['section'],
                    'mean_3y': datos_pais['num_patentes'].tail(3).mean(),
                    'std_3y': datos_pais['num_patentes'].tail(3).std(),
                    'growth_3y': datos_pais['growth_rate'].tail(3).mean() if 'growth_rate' in datos_pais.columns else 0.03
                }
                
                # Crear DataFrame de características
                df_features = pd.DataFrame([features])
                
                # Asegurar todas las columnas
                for col in self.feature_columns:
                    if col not in df_features.columns:
                        df_features[col] = 0
                
                # Escalar y predecir
                X_pred = df_features[self.feature_columns]
                X_pred_scaled = self.scaler.transform(X_pred)
                
                # Predicción ensemble ponderada
                pred_rf = self.models['random_forest'].predict(X_pred_scaled)[0]
                pred_gb = self.models['gradient_boosting'].predict(X_pred_scaled)[0]
                pred_ensemble = pred_rf * self.ensemble_weights['random_forest'] + \
                               pred_gb * self.ensemble_weights['gradient_boosting']
                
                # Asegurar predicción no negativa
                pred_final = max(10, pred_ensemble)
                
                # Determinar tendencia
                if año_offset == 1:
                    tendencia = 'crecimiento' if pred_final > ultimos_datos['num_patentes'] else 'decrecimiento'
                else:
                    pred_anterior = next((p['prediccion_patentes'] for p in predicciones 
                                        if p['pais'] == pais and p['año'] == año_futuro-1), ultimos_datos['num_patentes'])
                    tendencia = 'crecimiento' if pred_final > pred_anterior else 'decrecimiento'
                
                predicciones.append({
                    'pais': pais,
                    'año': año_futuro,
                    'años_desde_2021': año_offset,
                    'prediccion_patentes': round(pred_final),
                    'tendencia': tendencia,
                    'confidence': 'alta' if año_offset <= 3 else 'media'
                })
        
        df_predicciones = pd.DataFrame(predicciones)
        
        return df_predicciones

def visualizar_predicciones_interactivas(df_predicciones):
    """Visualizaciones interactivas para predicciones"""
    
    # 1. Gráfico de barras para 2031
    pred_2031 = df_predicciones[df_predicciones['año'] == 2031].sort_values('prediccion_patentes', ascending=False).head(10)
    
    fig1 = px.bar(
        pred_2031,
        x='prediccion_patentes',
        y='pais',
        orientation='h',
        color='tendencia',
        color_discrete_map={'crecimiento': 'green', 'decrecimiento': 'red'},
        title='Top 10 Países - Predicción 2031',
        labels={'prediccion_patentes': 'Patentes Predichas', 'pais': 'País'}
    )
    
    fig1.update_layout(height=400)
    
    # 2. Evolución temporal para top 3 países
    top_3_paises = pred_2031['pais'].head(3).tolist()
    
    fig2 = go.Figure()
    
    for pais in top_3_paises:
        datos_pais = df_predicciones[df_predicciones['pais'] == pais].sort_values('año')
        
        fig2.add_trace(go.Scatter(
            x=datos_pais['año'],
            y=datos_pais['prediccion_patentes'],
            mode='lines+markers',
            name=pais,
            line=dict(width=2)
        ))
    
    fig2.update_layout(
        title='Evolución de Predicciones 2022-2031 - Top 3 Países',
        xaxis_title='Año',
        yaxis_title='Patentes Predichas',
        height=400,
        hovermode='x unified'
    )
    
    return fig1, fig2

# ============================================================================
# 4. APLICACIÓN PRINCIPAL STREAMLIT
# ============================================================================

def main():
    """Función principal de la aplicación Streamlit"""
    
    # Sidebar
    with st.sidebar:
        st.title("USPTO Patent Analysis")
        st.markdown("---")
        
        st.subheader("⚙️ Configuración")
        
        # Selector de modo de datos
        modo_datos = st.radio(
            "Fuente de datos:",
            ["Datos de Ejemplo", "Google Cloud Storage"],
            index=0  # Por defecto datos de ejemplo
        )
        
        # Botón para cargar datos
        if st.button("🔄 Cargar Datos", type="primary", use_container_width=True):
            if modo_datos == "Google Cloud Storage":
                if not GCS_AVAILABLE:
                    st.error("❌ google-cloud-storage no está instalado. Instala con: pip install google-cloud-storage")
                else:
                    with st.spinner("Cargando datos desde GCS..."):
                        df_original = cargar_datos_desde_gcs()
                        df_original, df_agregado = preparar_datos_visualizacion(df_original)
                        
                        # Guardar en session state
                        st.session_state['df_original'] = df_original
                        st.session_state['df_agregado'] = df_agregado
                        
                        st.success("✅ Datos cargados exitosamente!")
            else:
                with st.spinner("Generando datos de ejemplo..."):
                    df_original = generar_datos_ejemplo()
                    df_original, df_agregado = preparar_datos_visualizacion(df_original)
                    
                    # Guardar en session state
                    st.session_state['df_original'] = df_original
                    st.session_state['df_agregado'] = df_agregado
                    
                    st.success("✅ Datos de ejemplo generados exitosamente!")
        
        st.markdown("---")
        st.subheader("📊 Navegación")
        
        # Menú de navegación
        pagina_seleccionada = st.radio(
            "Selecciona una sección:",
            [
                "🏠 Inicio",
                "📈 Análisis Histórico", 
                "🤖 Predicción ML",
                "🔍 Análisis por País"
            ]
        )
        
        st.markdown("---")
        st.info("💡 **Nota:** Usa 'Datos de Ejemplo' para pruebas rápidas.")
    
    # Contenido principal
    if pagina_seleccionada == "🏠 Inicio":
        st.header("🏠 Bienvenido al Sistema de Análisis de Patentes USPTO")
        
        st.markdown("""
        ### 🌟 Características Principales
        
        Este sistema integrado ofrece:
        
        1. **📊 Análisis Histórico Completo**
           - Visualización de datos de patentes desde 2006
           - Mapas interactivos por país y año
           - Análisis por sección tecnológica
        
        2. **🤖 Predicción con Machine Learning**
           - Modelos Ensemble Learning (Random Forest + Gradient Boosting)
           - Predicción de patentes 2022-2031
           - Métricas de evaluación detalladas
           - Visualización de predicciones futuras
        
        3. **🔍 Herramientas de Análisis**
           - Análisis detallado por país
           - Comparativas y tendencias
           - Filtros interactivos
        
        ### 🚀 Cómo Empezar
        
        1. Haz clic en **"Cargar Datos"** en la barra lateral
        2. Selecciona la fuente de datos
        3. Navega por las diferentes secciones usando el menú
        4. Explora las visualizaciones y predicciones
        
        """)
        
        # Mostrar estadísticas rápidas si hay datos cargados
        if 'df_agregado' in st.session_state:
            st.subheader("📊 Vista Rápida de Datos")
            
            df_agregado = st.session_state['df_agregado']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Período",
                    value=f"{df_agregado['year'].min()}-{df_agregado['year'].max()}"
                )
            
            with col2:
                st.metric(
                    label="Países",
                    value=df_agregado['assignee_country'].nunique()
                )
            
            with col3:
                st.metric(
                    label="Patentes Totales",
                    value=f"{df_agregado['num_patentes'].sum():,.0f}"
                )
    
    elif pagina_seleccionada == "📈 Análisis Histórico":
        st.header("📈 Análisis Histórico de Patentes")
        
        if 'df_agregado' not in st.session_state or 'df_original' not in st.session_state:
            st.warning("⚠ Por favor carga los datos primero desde la barra lateral.")
            return
        
        df_agregado = st.session_state['df_agregado']
        df_original = st.session_state['df_original']
        
        # Pestañas para diferentes tipos de análisis
        tab1, tab2, tab3 = st.tabs([
            "🌍 Mapa Mundial", 
            "📈 Tendencias", 
            "📋 Estadísticas"
        ])
        
        with tab1:
            st.subheader("Mapa Mundial Interactivo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                year_filtro = st.selectbox(
                    "Selecciona el año:",
                    options=['Todos'] + sorted(df_agregado['year'].unique().tolist()),
                    index=0
                )
            
            with col2:
                if 'section' in df_original.columns:
                    secciones_validas = df_original['section'].dropna().astype(str).unique().tolist()
                    section_filtro = st.selectbox(
                        "Sección tecnológica:",
                        options=['Todas'] + sorted(secciones_validas),
                        index=0
                    )
                else:
                    section_filtro = 'Todas'
            
            # Generar mapa
            year = None if year_filtro == 'Todos' else int(year_filtro)
            section = None if section_filtro == 'Todas' else section_filtro
            
            fig_mapa = crear_mapa_mundial_interactivo(df_agregado, year, section, df_original)
            st.plotly_chart(fig_mapa, use_container_width=True)
        
        with tab2:
            st.subheader("Tendencias Anuales")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pais_filtro = st.selectbox(
                    "País (opcional):",
                    options=['Todos'] + sorted(df_agregado['assignee_country'].unique().tolist()),
                    index=0
                )
            
            with col2:
                if 'section' in df_original.columns:
                    seccion_filtro = st.selectbox(
                        "Sección (opcional):",
                        options=['Todas'] + sorted(df_original['section'].unique().tolist()),
                        index=0,
                        key="seccion_tendencia"
                    )
                else:
                    seccion_filtro = 'Todas'
            
            # Generar gráfico
            pais = None if pais_filtro == 'Todos' else pais_filtro
            seccion = None if seccion_filtro == 'Todas' else seccion_filtro
            
            fig_tendencia = grafico_tendencia_anual_interactivo(df_agregado, pais, seccion, df_original)
            st.plotly_chart(fig_tendencia, use_container_width=True)
        
        with tab3:
            st.subheader("Estadísticas Detalladas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Período Histórico",
                    value=f"{df_agregado['year'].min()} - {df_agregado['year'].max()}"
                )
                
                st.metric(
                    label="Países Únicos",
                    value=df_agregado['assignee_country'].nunique()
                )
            
            with col2:
                st.metric(
                    label="Total Patentes",
                    value=f"{df_agregado['num_patentes'].sum():,.0f}"
                )
                
                st.metric(
                    label="Patentes/Año Promedio",
                    value=f"{df_agregado.groupby('year')['num_patentes'].sum().mean():,.0f}"
                )
            
            with col3:
                if 'section' in df_original.columns:
                    st.metric(
                        label="Secciones Únicas",
                        value=df_original['section'].nunique()
                    )
                
                # País líder
                pais_lider = df_agregado.groupby('assignee_country')['num_patentes'].sum().idxmax()
                patentes_lider = int(df_agregado.groupby('assignee_country')['num_patentes'].sum().max())
                st.metric(
                    label="País Líder",
                    value=f"{pais_lider} ({patentes_lider:,})"
                )
            
            # Top 10 países
            st.subheader("🏆 Top 10 Países por Patentes Totales")
            
            top_paises = df_agregado.groupby('assignee_country')['num_patentes'].sum().nlargest(10)
            
            # Crear DataFrame para visualización
            top_df = pd.DataFrame({
                'País': top_paises.index,
                'Patentes': top_paises.values,
                'Porcentaje': (top_paises.values / top_paises.values.sum() * 100)
            })
            
            # Mostrar tabla
            st.dataframe(
                top_df.style.format({
                    'Patentes': '{:,.0f}',
                    'Porcentaje': '{:.1f}%'
                }),
                use_container_width=True
            )
    
    elif pagina_seleccionada == "🤖 Predicción ML":
        st.header("🤖 Predicción con Ensemble Learning")
        
        if 'df_agregado' not in st.session_state:
            st.warning("⚠ Por favor carga los datos primero desde la barra lateral.")
            return
        
        df_agregado = st.session_state['df_agregado']
        
        tab1, tab2 = st.tabs([
            "🔮 Entrenar Modelos", 
            "📊 Ver Predicciones"
        ])
        
        with tab1:
            st.subheader("Entrenamiento de Modelos Ensemble")
            
            col1, col2 = st.columns(2)
            
            with col1:
                horizonte_prediccion = st.slider(
                    "Horizonte de predicción (años):",
                    min_value=1,
                    max_value=10,
                    value=6
                )
            
            with col2:
                num_paises_prediccion = st.slider(
                    "Número de países a predecir:",
                    min_value=5,
                    max_value=15,
                    value=10
                )
            
            if st.button("🚀 Entrenar Modelos y Generar Predicciones", type="primary", use_container_width=True):
                
                with st.spinner("Entrenando modelos..."):
                    # Inicializar predictor
                    ensemble_model = EnsemblePredictorStreamlit()
                    
                    # Preparar datos
                    datos_preparados = ensemble_model.preparar_datos_para_prediccion(df_agregado)
                    
                    # Crear dataset de entrenamiento
                    df_entrenamiento = ensemble_model.crear_dataset_entrenamiento(datos_preparados, horizonte_prediccion)
                    
                    # Entrenar modelos
                    X_test, y_test, resultados = ensemble_model.entrenar_modelos(df_entrenamiento, horizonte_prediccion)
                    
                    # Generar predicciones futuras
                    top_paises = df_agregado.groupby('assignee_country')['num_patentes'].sum().nlargest(num_paises_prediccion).index
                    df_predicciones = ensemble_model.predecir_futuro(
                        df_agregado, años_futuros=10, países_interes=top_paises.tolist()
                    )
                    
                    # Guardar en session state
                    st.session_state['ensemble_model'] = ensemble_model
                    st.session_state['df_predicciones'] = df_predicciones
                    st.session_state['resultados_entrenamiento'] = resultados
                    
                    st.success("✅ ¡Modelos entrenados y predicciones generadas exitosamente!")
                    
                    # Mostrar métricas rápidas
                    if ensemble_model.metrics:
                        st.subheader("📊 Métricas del Modelo")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            rf_metrics = ensemble_model.metrics.get('random_forest', {})
                            st.metric("Random Forest R²", f"{rf_metrics.get('test_r2', 0):.3f}")
                            st.metric("Random Forest MAE", f"{rf_metrics.get('test_mae', 0):.2f}")
                        
                        with col2:
                            gb_metrics = ensemble_model.metrics.get('gradient_boosting', {})
                            st.metric("Gradient Boosting R²", f"{gb_metrics.get('test_r2', 0):.3f}")
                            st.metric("Gradient Boosting MAE", f"{gb_metrics.get('test_mae', 0):.2f}")
        
        with tab2:
            st.subheader("Visualización de Predicciones")
            
            if 'df_predicciones' not in st.session_state:
                st.info("ℹ️ Primero entrena los modelos en la pestaña 'Entrenar Modelos'.")
            else:
                df_predicciones = st.session_state['df_predicciones']
                
                # Mostrar resumen
                pred_2031 = df_predicciones[df_predicciones['año'] == 2031].sort_values('prediccion_patentes', ascending=False)
                
                st.subheader("📊 Resumen de Predicciones 2031")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total Predicho",
                        value=f"{int(pred_2031['prediccion_patentes'].sum()):,}"
                    )
                
                with col2:
                    st.metric(
                        label="Países con Crecimiento",
                        value=f"{(pred_2031['tendencia'] == 'crecimiento').sum()}"
                    )
                
                with col3:
                    st.metric(
                        label="Países Analizados",
                        value=f"{df_predicciones['pais'].nunique()}"
                    )
                
                # Mostrar visualizaciones
                fig1, fig2 = visualizar_predicciones_interactivas(df_predicciones)
                
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Tabla detallada de predicciones
                st.subheader("📋 Datos Detallados de Predicciones")
                
                año_filtro = st.selectbox(
                    "Filtrar por año:",
                    options=['Todos'] + sorted(df_predicciones['año'].unique().tolist()),
                    index=0
                )
                
                if año_filtro == 'Todos':
                    datos_filtrados = df_predicciones
                else:
                    datos_filtrados = df_predicciones[df_predicciones['año'] == int(año_filtro)]
                
                st.dataframe(
                    datos_filtrados.sort_values(['año', 'prediccion_patentes'], ascending=[True, False]).rename(
                        columns={
                            'pais': 'País',
                            'año': 'Año',
                            'prediccion_patentes': 'Patentes Predichas',
                            'tendencia': 'Tendencia',
                            'confidence': 'Confianza'
                        }
                    ).style.format({
                        'Patentes Predichas': '{:,.0f}'
                    }),
                    use_container_width=True
                )
    
    elif pagina_seleccionada == "🔍 Análisis por País":
        st.header("🔍 Análisis Detallado por País")
        
        if 'df_agregado' not in st.session_state:
            st.warning("⚠ Por favor carga los datos primero desde la barra lateral.")
            return
        
        df_agregado = st.session_state['df_agregado']
        df_predicciones = st.session_state.get('df_predicciones', None)
        
        st.subheader("🇺🇸 Analizar País Específico")
        
        paises_disponibles = sorted(df_agregado['assignee_country'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            pais_seleccionado = st.selectbox(
                "Selecciona un país:",
                paises_disponibles,
                index=0 if 'US' in paises_disponibles else 0
            )
        
        with col2:
            if df_predicciones is not None:
                pred_2031_pais = df_predicciones[(df_predicciones['pais'] == pais_seleccionado) & 
                                               (df_predicciones['año'] == 2031)]
                if not pred_2031_pais.empty:
                    st.metric(
                        label="Predicción 2031",
                        value=f"{int(pred_2031_pais['prediccion_patentes'].iloc[0]):,}"
                    )
        
        if pais_seleccionado:
            # Obtener datos del país
            datos_pais = df_agregado[df_agregado['assignee_country'] == pais_seleccionado].sort_values('year')
            
            if not datos_pais.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Período",
                        value=f"{datos_pais['year'].min()} - {datos_pais['year'].max()}"
                    )
                
                with col2:
                    st.metric(
                        label="Total Patentes",
                        value=f"{int(datos_pais['num_patentes'].sum()):,}"
                    )
                
                with col3:
                    crecimiento = ((datos_pais['num_patentes'].iloc[-1] - datos_pais['num_patentes'].iloc[0]) / 
                                 datos_pais['num_patentes'].iloc[0] * 100) if len(datos_pais) > 1 else 0
                    st.metric(
                        label="Crecimiento Histórico",
                        value=f"{crecimiento:.1f}%"
                    )
                
                # Gráfico de evolución
                fig = px.line(
                    datos_pais,
                    x='year',
                    y='num_patentes',
                    title=f'Evolución de Patentes - {pais_seleccionado}',
                    markers=True
                )
                
                # Añadir predicciones si están disponibles
                if df_predicciones is not None:
                    pred_pais = df_predicciones[df_predicciones['pais'] == pais_seleccionado].sort_values('año')
                    
                    if not pred_pais.empty:
                        fig.add_trace(go.Scatter(
                            x=pred_pais['año'],
                            y=pred_pais['prediccion_patentes'],
                            mode='lines+markers',
                            name='Predicciones',
                            line=dict(dash='dash', color='red')
                        ))
                
                fig.update_layout(
                    xaxis_title="Año",
                    yaxis_title="Patentes",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# EJECUCIÓN DE LA APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    # Inicializar session state si no existe
    if 'datos_cargados' not in st.session_state:
        st.session_state['datos_cargados'] = False
    
    # Ejecutar aplicación
    main()




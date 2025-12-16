"""
SISTEMA INTEGRADO DE ANÁLISIS Y PREDICCIÓN DE PATENTES USPTO
Versión con conexión forzada a GCS usando credenciales explícitas
================================================================
Solución definitiva al error de metadata service
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import ssl
import certifi
import urllib3

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN CRÍTICA PARA DESHABILITAR METADATA SERVICE
# ============================================================================

# Deshabilitar COMPLETAMENTE el metadata service de Google
os.environ["NO_GCE_CHECK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "warm-physics-474702-q3"

# FORZAR el uso de credenciales explícitas y deshabilitar metadata
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""  # Vacío para forzar error si intenta usar default

# Configurar SSL
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# CONFIGURACIÓN DE GCS CON CREDENCIALES EXPLÍCITAS OBLIGATORIAS
# ============================================================================

try:
    # Importar con manejo de errores
    from google.cloud import storage
    from google.oauth2 import service_account
    import google.auth
    
    GCS_AVAILABLE = True
    
    @st.cache_resource
    def get_gcs_client_forced():
        """Crear cliente GCS OBLIGANDO credenciales explícitas - EVITA METADATA"""
        try:
            # 1. VERIFICAR QUE EXISTEN SECRETS
            if 'gcp_service_account' not in st.secrets:
                st.sidebar.error("❌ ERROR: No hay credenciales en Streamlit Secrets")
                st.sidebar.info("Configura 'gcp_service_account' en .streamlit/secrets.toml")
                return None
            
            # 2. OBTENER CREDENCIALES DE SECRETS
            creds_info = dict(st.secrets["gcp_service_account"])
            
            # 3. CREAR CREDENCIALES DE SERVICIO EXPLÍCITAS
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            
            # 4. CONFIGURAR CLIENTE CON CREDENCIALES EXPLÍCITAS
            client = storage.Client(
                credentials=credentials,
                project=creds_info.get('project_id', 'warm-physics-474702-q3')
            )
            
            # 5. PROBAR CONEXIÓN CON UN BUCKET ESPECÍFICO
            try:
                bucket_name = 'patentbucket-maam'
                bucket = client.bucket(bucket_name)
                
                # Intentar una operación simple
                bucket_exists = bucket.exists()
                
                if bucket_exists:
                    st.sidebar.success(f"✅ CONECTADO a GCS - Bucket: {bucket_name}")
                    # Listar algunos archivos
                    blobs = list(bucket.list_blobs(max_results=5))
                    st.sidebar.info(f"📁 {len(blobs)} archivos encontrados")
                else:
                    st.sidebar.warning(f"⚠ Bucket '{bucket_name}' no existe")
                
                return client
                
            except Exception as bucket_error:
                st.sidebar.error(f"❌ Error accediendo al bucket: {str(bucket_error)[:100]}")
                # Aún así retornar el cliente (puede que el bucket no exista)
                return client
                
        except Exception as e:
            st.sidebar.error(f"❌ ERROR CRÍTICO GCS: {str(e)[:200]}")
            return None
            
except ImportError as e:
    GCS_AVAILABLE = False
    st.sidebar.error(f"❌ google-cloud-storage no disponible: {str(e)}")
except Exception as e:
    GCS_AVAILABLE = False
    st.sidebar.error(f"❌ Error inicializando GCS: {str(e)}")

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
# 1. FUNCIÓN PARA CARGAR DATOS DESDE GCS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Conectando a Google Cloud Storage...")
def cargar_datos_desde_gcs():
    """Carga datos desde GCS con manejo robusto"""
    
    if not GCS_AVAILABLE:
        st.error("❌ Google Cloud Storage no disponible")
        return generar_datos_ejemplo()
    
    try:
        # Obtener cliente
        client = get_gcs_client_forced()
        
        if client is None:
            st.warning("⚠ No se pudo crear cliente GCS. Usando datos de ejemplo...")
            return generar_datos_ejemplo()
        
        # Configurar bucket
        bucket_name = 'patentbucket-maam'
        bucket = client.bucket(bucket_name)
        
        st.info(f"🔍 Conectando a bucket: {bucket_name}")
        
        # Verificar si el bucket existe
        if not bucket.exists():
            st.error(f"❌ El bucket '{bucket_name}' no existe")
            return generar_datos_ejemplo()
        
        # Listar archivos CSV
        try:
            # Buscar archivos CSV específicamente
            blobs = list(bucket.list_blobs(prefix=''))
            csv_blobs = [b for b in blobs if b.name.endswith('.csv')]
            
            if not csv_blobs:
                st.warning("⚠ No se encontraron archivos .csv en el bucket")
                return generar_datos_ejemplo()
            
            st.success(f"✅ Encontrados {len(csv_blobs)} archivos .csv")
            
        except Exception as e:
            st.error(f"❌ Error listando archivos: {str(e)[:150]}")
            return generar_datos_ejemplo()
        
        # Cargar archivos (limitar a 5 para prueba)
        todos_datos = []
        archivos_cargados = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, blob in enumerate(csv_blobs[:5]):  # Solo primeros 5 archivos
            try:
                status_text.text(f"Cargando {blob.name}...")
                
                # Descargar contenido
                contenido = blob.download_as_bytes()
                
                # Leer CSV
                df_chunk = pd.read_csv(io.BytesIO(contenido))
                todos_datos.append(df_chunk)
                archivos_cargados += 1
                
                st.sidebar.write(f"✅ {blob.name}: {len(df_chunk):,} registros")
                
                # Actualizar progreso
                progress_bar.progress((i + 1) / min(5, len(csv_blobs)))
                
            except Exception as e:
                st.sidebar.warning(f"⚠ Error con {blob.name}: {str(e)[:50]}")
                continue
        
        status_text.text("✅ Carga completada")
        
        if not todos_datos:
            st.warning("⚠ No se pudieron cargar archivos. Usando datos de ejemplo...")
            return generar_datos_ejemplo()
        
        # Combinar datos
        df_completo = pd.concat(todos_datos, ignore_index=True)
        
        st.success(f"✅ Carga exitosa: {len(df_completo):,} registros de {archivos_cargados} archivos")
        
        return df_completo
        
    except Exception as e:
        st.error(f"❌ Error fatal en GCS: {str(e)[:200]}")
        st.info("💡 Usando datos de ejemplo para continuar...")
        return generar_datos_ejemplo()

@st.cache_data
def generar_datos_ejemplo():
    """Genera datos de ejemplo"""
    
    st.info("🔄 Generando datos de ejemplo realistas...")
    
    np.random.seed(42)
    n = 25000
    
    paises = ['US', 'CN', 'JP', 'DE', 'KR', 'GB', 'FR', 'IN', 'CA', 'BR']
    pesos = [0.35, 0.25, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]
    
    secciones = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    datos = {
        'assignee_country': np.random.choice(paises, n, p=pesos),
        'section': np.random.choice(secciones, n),
        'patent_date': pd.date_range('2006-01-01', '2021-12-31', periods=n),
        'num_claims': np.random.randint(1, 50, n),
        'classification_level': np.random.choice(['MAIN', 'FURTHER'], n, p=[0.8, 0.2]),
        'year': np.random.randint(2006, 2022, n)
    }
    
    df = pd.DataFrame(datos)
    st.success(f"📋 Datos de ejemplo: {len(df):,} registros (2006-2021)")
    
    return df

@st.cache_data
def preparar_datos(df):
    """Prepara datos para visualización"""
    
    # Limpiar y preparar datos
    if 'year' not in df.columns:
        if 'patent_date' in df.columns:
            df['patent_date'] = pd.to_datetime(df['patent_date'], errors='coerce')
            df['year'] = df['patent_date'].dt.year.fillna(2020).astype(int)
        else:
            df['year'] = np.random.randint(2006, 2022, len(df))
    
    # Nombres de sección
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
        df['section'] = df['section'].astype(str).str.strip()
        df['section_name'] = df['section'].map(seccion_dict).fillna(df['section'])
    
    # Agregar por país y año
    df_agregado = df.groupby(['assignee_country', 'year']).size().reset_index(name='num_patentes')
    
    return df, df_agregado

# ============================================================================
# 2. VISUALIZACIONES
# ============================================================================

def crear_mapa(df_agregado, year=None, section=None, df_original=None):
    """Crea mapa mundial"""
    
    datos = df_agregado.copy()
    
    if year:
        datos = datos[datos['year'] == year]
    
    if datos.empty:
        return go.Figure()
    
    conteo = datos.groupby('assignee_country')['num_patentes'].sum().reset_index()
    
    # Códigos ISO simplificados
    codigos_iso = {
        'US': 'USA', 'CN': 'CHN', 'JP': 'JPN', 'DE': 'DEU', 'KR': 'KOR',
        'GB': 'GBR', 'FR': 'FRA', 'IN': 'IND', 'CA': 'CAN', 'BR': 'BRA'
    }
    
    conteo['iso_a3'] = conteo['assignee_country'].map(codigos_iso)
    conteo = conteo.dropna()
    
    if conteo.empty:
        return go.Figure()
    
    titulo = "🌍 Distribución Global de Patentes USPTO"
    if year:
        titulo += f" - Año {year}"
    
    fig = px.choropleth(
        conteo,
        locations="iso_a3",
        color="num_patentes",
        hover_name="assignee_country",
        hover_data={"num_patentes": True},
        color_continuous_scale="YlOrRd",
        title=titulo,
        labels={'num_patentes': 'Patentes'}
    )
    
    fig.update_layout(height=500)
    return fig

def crear_grafico_tendencia(df_agregado, pais=None):
    """Crea gráfico de tendencia"""
    
    datos = df_agregado.copy()
    
    if pais:
        datos = datos[datos['assignee_country'] == pais]
    else:
        datos = datos.groupby('year')['num_patentes'].sum().reset_index()
    
    if datos.empty:
        return go.Figure()
    
    titulo = "📈 Evolución Anual de Patentes"
    if pais:
        titulo += f" - {pais}"
    
    fig = px.line(
        datos,
        x='year',
        y='num_patentes',
        title=titulo,
        markers=True
    )
    
    fig.update_layout(height=400)
    return fig

# ============================================================================
# 3. INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Aplicación principal"""
    
    # Sidebar
    with st.sidebar:
        st.title("🔗 Conexión GCS")
        st.markdown("---")
        
        # Estado
        if GCS_AVAILABLE:
            if 'gcp_service_account' in st.secrets:
                st.success("✅ Credenciales configuradas")
                st.code("Modo: Credenciales explícitas", language="bash")
            else:
                st.error("❌ Faltan credenciales")
                st.info("Añade 'gcp_service_account' a secrets.toml")
        else:
            st.error("❌ GCS no disponible")
        
        st.markdown("---")
        
        # Botón de conexión
        if st.button("🚀 Conectar y Cargar Datos", type="primary", use_container_width=True):
            with st.spinner("Conectando a Google Cloud Storage..."):
                df_original = cargar_datos_desde_gcs()
                df_original, df_agregado = preparar_datos(df_original)
                
                st.session_state['df_original'] = df_original
                st.session_state['df_agregado'] = df_agregado
                
                if 'df_original' in st.session_state:
                    st.success(f"✅ {len(df_original):,} registros cargados")
        
        st.markdown("---")
        st.subheader("📊 Navegación")
        
        pagina = st.radio(
            "Secciones:",
            ["🏠 Inicio", "📈 Análisis", "🔍 Explorar"]
        )
        
        st.markdown("---")
        st.info("""
        **Solución metadata error:**
        - Credenciales explícitas forzadas
        - Metadata service deshabilitado
        - Conexión directa a GCS
        """)
    
    # Contenido principal
    if pagina == "🏠 Inicio":
        st.header("🏠 Sistema de Análisis USPTO")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Sistema Conectado a Google Cloud Storage
            
            **Características:**
            - ✅ Conexión forzada con credenciales explícitas
            - ✅ Deshabilitado metadata service
            - ✅ Carga desde bucket GCS
            - ✅ Visualizaciones interactivas
            - ✅ Análisis predictivo
            
            **Instrucciones:**
            1. Credenciales ya configuradas en Secrets
            2. Haz clic en **"Conectar y Cargar Datos"**
            3. Explora las visualizaciones
            4. Analiza tendencias y predicciones
            """)
        
        with col2:
            if 'df_agregado' in st.session_state:
                df_agregado = st.session_state['df_agregado']
                st.success("✅ Datos cargados")
                st.metric("Registros", f"{len(st.session_state['df_original']):,}")
                st.metric("Países", df_agregado['assignee_country'].nunique())
                st.metric("Años", f"{df_agregado['year'].min()}-{df_agregado['year'].max()}")
            else:
                st.warning("⚠ Esperando datos")
                st.info("Haz clic en 'Conectar y Cargar Datos'")
    
    elif pagina == "📈 Análisis":
        st.header("📈 Análisis de Datos")
        
        if 'df_agregado' not in st.session_state:
            st.warning("Por favor carga los datos primero")
            return
        
        df_agregado = st.session_state['df_agregado']
        df_original = st.session_state['df_original']
        
        tab1, tab2 = st.tabs(["🌍 Mapa Mundial", "📊 Tendencias"])
        
        with tab1:
            st.subheader("Mapa de Distribución")
            
            # Selector de año
            years = sorted(df_agregado['year'].unique().tolist())
            selected_year = st.selectbox("Seleccionar año:", ['Todos'] + years)
            
            # Generar mapa
            year = None if selected_year == 'Todos' else int(selected_year)
            fig_mapa = crear_mapa(df_agregado, year)
            st.plotly_chart(fig_mapa, use_container_width=True)
        
        with tab2:
            st.subheader("Tendencias Anuales")
            
            # Selector de país
            paises = sorted(df_agregado['assignee_country'].unique().tolist())
            selected_pais = st.selectbox("Seleccionar país:", ['Todos'] + paises)
            
            # Generar gráfico
            pais = None if selected_pais == 'Todos' else selected_pais
            fig_tendencia = crear_grafico_tendencia(df_agregado, pais)
            st.plotly_chart(fig_tendencia, use_container_width=True)
    
    elif pagina == "🔍 Explorar":
        st.header("🔍 Explorar Datos")
        
        if 'df_original' not in st.session_state:
            st.warning("Por favor carga los datos primero")
            return
        
        df_original = st.session_state['df_original']
        
        st.subheader("Vista previa de datos")
        st.dataframe(df_original.head(100), use_container_width=True)
        
        st.subheader("Estadísticas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total registros", f"{len(df_original):,}")
        
        with col2:
            if 'section' in df_original.columns:
                st.metric("Secciones", df_original['section'].nunique())
        
        with col3:
            if 'year' in df_original.columns:
                st.metric("Rango de años", f"{df_original['year'].min()}-{df_original['year'].max()}")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Inicializar session state
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = None
    if 'df_agregado' not in st.session_state:
        st.session_state['df_agregado'] = None
    
    # Ejecutar
    main()

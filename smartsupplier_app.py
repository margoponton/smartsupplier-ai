# SmartSupplier AI | GP Intelligence
# Autor: Ines Margarita Gonzalez Ponton

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
from datetime import datetime
import tensorflow as tf

st.set_page_config(
    page_title="SmartSupplier AI | GP Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Fondo general azul claro oscuro */
.stApp {
    background-color: #0d1b2a;
    color: #e8e8f0;
}

/* Sidebar morado oscuro */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1040 0%, #0d0d2b 100%);
    border-right: 1px solid #423557;
}

/* Hero header morado */
.hero-header {
    background: linear-gradient(135deg, #1a1040 0%, #2d1b69 50%, #1a1040 100%);
    border: 1px solid #600EF6;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.hero-badge {
    display: inline-block;
    background: #600EF6;
    color: white;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
    font-weight: 500;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: #a0a0c0;
    margin: 6px 0 0;
}

/* Cards de riesgo */
.riesgo-alto  { background: rgba(220,38,38,0.12); border-left: 4px solid #dc2626; padding: 16px 20px; border-radius: 8px; margin: 12px 0; }
.riesgo-medio { background: rgba(234,179,8,0.12);  border-left: 4px solid #eab308; padding: 16px 20px; border-radius: 8px; margin: 12px 0; }
.riesgo-bajo  { background: rgba(34,197,94,0.12);  border-left: 4px solid #22c55e; padding: 16px 20px; border-radius: 8px; margin: 12px 0; }

/* Pilar cards */
.pilar-card {
    background: rgba(13,27,42,0.8);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.pilar-title { color: #4da6ff; font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; }
.pilar-text  { color: #b0c4d8; font-size: 0.85rem; line-height: 1.6; }

/* Steps */
.step-card {
    background: rgba(13,27,42,0.8);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.step-num   { color: #600EF6; font-weight: 700; }
.step-title { color: #e8e8f0; font-weight: 600; font-size: 0.9rem; }
.step-desc  { color: #b0c4d8; font-size: 0.82rem; line-height: 1.5; }

/* Footer */
.footer-gp {
    background: rgba(13,27,42,0.9);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 40px;
    text-align: center;
}
.footer-name    { color: #4da6ff; font-weight: 600; font-size: 1rem; }
.footer-role    { color: #a0a0c0; font-size: 0.82rem; margin: 4px 0; }
.footer-contact { color: #b0c4d8; font-size: 0.8rem; line-height: 1.8; }
.footer-brand   { color: #666680; font-size: 0.75rem; margin-top: 10px; }

.gp-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #600EF6, transparent);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DEL MODELO
# ============================================================
@st.cache_resource
def cargar_modelo():
    try:
        model     = tf.keras.models.load_model('models_saved/smartprocess_model.keras')
        scaler    = joblib.load('models_saved/scaler.pkl')
        features  = joblib.load('models_saved/features.pkl')
        label_map = joblib.load('models_saved/label_map.pkl')
        with open('models_saved/config_api.json') as f:
            config = json.load(f)
        return model, scaler, features, label_map, config
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.stop()

model, scaler, FEATURES, LABEL_MAP_INV, config = cargar_modelo()
COLORES = {'Alto': '#dc2626', 'Medio': '#eab308', 'Bajo': '#22c55e'}

# ============================================================
# ESTADO DE SESION
# ============================================================
if 'evaluaciones' not in st.session_state:
    st.session_state.evaluaciones = []
if 'empresa' not in st.session_state:
    st.session_state.empresa = ''
if 'area' not in st.session_state:
    st.session_state.area = ''
if 'form_count' not in st.session_state:
    st.session_state.form_count = 0

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    try:
        st.image('logo_gp.png', width=180)
    except:
        st.markdown("**GP Intelligence**")
        st.caption("Engineering Intelligent Decisions")

    st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
    st.markdown("**SmartSupplier AI**")
    st.caption("Clasificacion de Riesgo de Proveedores")
    st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)

    st.markdown("**Informacion de la empresa**")
    empresa = st.text_input("Nombre de tu empresa", placeholder="Ej: Grupo Nutresa")
    area    = st.text_input("Area / Departamento",   placeholder="Ej: Compras, Logistica")
    correo   = st.text_input("Correo",               placeholder="Ej: compras@empresa.com")
    web      = st.text_input("Pagina web",            placeholder="Ej: www.empresa.com")

    if empresa: st.session_state.empresa = empresa
    if area:    st.session_state.area    = area
    

    st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
    st.metric("Evaluaciones en sesion", len(st.session_state.evaluaciones))

    if st.session_state.evaluaciones:
        altos  = sum(1 for e in st.session_state.evaluaciones if e['Riesgo'] == 'Alto')
        medios = sum(1 for e in st.session_state.evaluaciones if e['Riesgo'] == 'Medio')
        bajos  = sum(1 for e in st.session_state.evaluaciones if e['Riesgo'] == 'Bajo')
        st.write(f"Alto: {altos} | Medio: {medios} | Bajo: {bajos}")

    st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
    st.caption("Modelo: Red Neuronal Keras\nF1 Weighted: 0.976\nv1.0 - 2025")

# ============================================================
# HERO HEADER
# ============================================================
empresa_actual = st.session_state.get('empresa', '')
area_actual    = st.session_state.get('area', '')
subtitulo_extra = f" | {empresa_actual} - {area_actual}" if empresa_actual and area_actual else (f" | {empresa_actual}" if empresa_actual else "")

st.markdown(f"""
<div class="hero-header">
    <span class="hero-badge">GP Intelligence | Deep Learning Project</span>
    <h1 class="hero-title">SmartSupplier AI</h1>
    <p class="hero-subtitle">Clasificacion predictiva de riesgo de proveedores. Engineering Intelligent Decisions{subtitulo_extra}.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab3, tab1, tab2 = st.tabs(["¿Como funciona?", "Evaluar Proveedor", "Historial de Sesión"])

# ============================================================
# TAB: COMO FUNCIONA
# ============================================================
with tab3:
    st.subheader("SmartSupplier AI")
    st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Decision de Diseño")
        st.markdown("""
        <div class="pilar-card">
            <div class="pilar-title">Por que se excluye Cumplimiento SLA</div>
            <div class="pilar-text">
                El campo Cumplimiento SLA se excluye intencionalmente de la variable objetivo
                porque SmartSupplier AI es un sistema predictivo: clasifica el riesgo ANTES
                de que ocurra la entrega. Cumplimiento se conoce solo post-entrega.<br><br>
                Validacion empirica: Con Cumplimiento F1 = 0.628. Sin Cumplimiento F1 = 0.976.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Marco Estrategico")
        pilares = [
            ("1. Identificacion de objetivos estrategicos",
             "Toda implementacion debe comenzar con la priorizacion de variables clave: lead time, costes logisticos, desviaciones contractuales o nivel de servicio."),
            ("2. Centralizacion y estructuracion de datos",
             "Es fundamental unificar las fuentes internas y externas (ERP, SRM, contratos, logistica) bajo modelos de datos limpios, consistentes y compatibles entre sistemas."),
            ("3. Seleccion tecnologica adecuada",
             "Las herramientas de analisis predictivo deben integrarse con la infraestructura digital existente, facilitando su conexion con modulos operativos y dashboards ejecutivos."),
            ("4. Formacion del equipo de compras",
             "El personal debe comprender como interpretar outputs predictivos, correlacionarlos con procesos diarios y activar decisiones automaticas o semi-automaticas.")
        ]
        for titulo, texto in pilares:
            st.markdown(f"""
            <div class="pilar-card">
                <div class="pilar-title">{titulo}</div>
                <div class="pilar-text">{texto}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Proceso de Generacion del Modelo")
        pasos = [
            ("Carga y exploracion (EDA)", "777 ordenes de compra reales. Analisis de nulos, distribuciones y relaciones entre variables."),
            ("Validacion de hipotesis", "ANOVA Cantidad vs Categoria (p=0.57, no significativo). Analisis de nulos por estado de orden."),
            ("Limpieza con logica de negocio", "Nulos imputados con mediana por proveedor. Sin eliminacion de filas: 777 registros conservados."),
            ("Feature Engineering", "4 features: defect_rate, savings_pct, cancelled_flag, delay_vs_median."),
            ("Variable objetivo sin leakage", "Clasificacion Alto/Medio/Bajo basada solo en factores pre-entrega. Cumplimiento excluido."),
            ("Balanceo con SMOTE", "Clase Alto con 32 casos. SMOTE k=3 balancea a 464 casos por clase."),
            ("Red neuronal Keras", "Dense(32) -> BN -> Dropout(0.3) -> Dense(16) -> BN -> Dropout(0.2) -> Dense(8) -> Softmax(3)."),
            ("Resultados", "F1 Weighted: 0.976 | Bajo: 0.99 | Medio: 0.93 | Alto: 0.73.")
        ]
        for i, (titulo, desc) in enumerate(pasos, 1):
            st.markdown(f"""
            <div class="step-card">
                <span class="step-num">{i}.</span>
                <span class="step-title"> {titulo}</span><br>
                <span class="step-desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
    st.markdown("#### Proyecto Educativo Escalable")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""
        <div class="pilar-card">
            <div class="pilar-title">Por que es educativo</div>
            <div class="pilar-text">
                SmartSupplier AI nace como proyecto final de un curso de IA y Deep Learning
                con criterios de ingenieria real: validacion estadistica, decision documentada
                sobre data leakage, balanceo SMOTE y pipeline modular exportable.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_e2:
        st.markdown("""
        <div class="pilar-card">
            <div class="pilar-title">Como y por que es escalable</div>
            <div class="pilar-text">
                El modelo se reentrena con datos de cualquier empresa reemplazando el CSV.
                SmartSupplier AI es la primera aplicacion bajo GP Intelligence, disenada
                para escalar a analisis de contratos, prediccion de demanda y optimizacion
                de lead times.
            </div>
        </div>
        """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("F1 Weighted", "0.976")
    mc2.metric("Clase Bajo",  "F1 0.99")
    mc3.metric("Clase Medio", "F1 0.93")
    mc4.metric("Clase Alto",  "F1 0.73")

# ============================================================
# TAB: EVALUAR PROVEEDOR
# ============================================================
with tab1:
    if not st.session_state.get('empresa'):
        st.warning("Ingresa el nombre de tu empresa en el panel izquierdo antes de evaluar.")

    st.subheader("Datos del Proveedor")
    st.caption("Completa los campos con el historial operativo del proveedor a evaluar.")

    fc = st.session_state.form_count

    col1, col2 = st.columns(2)

    with col1:
        nombre_proveedor     = st.text_input("Nombre del proveedor *",
                                            placeholder="Ej: Alpha Inc, Delta Logistics",
                                              key=f"nombre_{fc}")
        categoria            = st.selectbox("Categoria del proveedor",
                                             ["Raw Materials","Packaging","MRO","Office Supplies","Otra"],
                                             key=f"cat_{fc}")
        cantidad             = st.number_input("Cantidad ordenada (unidades)",
                                               min_value=1, value=1000, step=100, key=f"cant_{fc}")
        unidades_defectuosas = st.number_input("Unidades defectuosas",
                                               min_value=0, value=0, step=1, key=f"defect_{fc}")
        nombre_producto = st.text_input("Nombre del producto / servicio",
                                                placeholder="Ej: Acero laminado, Empaques PET", key=f"producto_{fc}")

    with col2:
        precio_unitario  = st.number_input("Precio unitario (USD)",
                                            min_value=0.01, value=50.0, step=0.5,
                                            format="%.2f", key=f"precio_{fc}")
        precio_negociado = st.number_input("Precio negociado (USD)",
                                            min_value=0.01, value=47.0, step=0.5,
                                            format="%.2f", key=f"negoc_{fc}")
        dias_entrega     = st.number_input("Dias reales de entrega",
                                            min_value=0, value=12, step=1, key=f"dias_{fc}")
        cancelada        = st.toggle("Orden cancelada?", value=False, key=f"cancel_{fc}")

    st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)

    with st.expander("Ver indicadores calculados", expanded=False):
        mediana_global   = config.get('mediana_global_delivery', 12.0)
        defect_rate_prev = unidades_defectuosas / cantidad if cantidad > 0 else 0
        savings_pct_prev = (precio_unitario - precio_negociado) / precio_unitario if precio_unitario > 0 else 0
        delay_prev       = dias_entrega - mediana_global if not cancelada else 0
        p1, p2, p3, p4  = st.columns(4)
        p1.metric("Tasa de defectos",   f"{defect_rate_prev:.1%}")
        p2.metric("Ahorro negociado",   f"{savings_pct_prev:.1%}")
        p3.metric("Retraso vs mediana", f"{delay_prev:+.1f} dias")
        p4.metric("Cancelada",          "Si" if cancelada else "No")

    col_b1, col_b2, col_b3 = st.columns([2, 2, 4])
    with col_b1:
        evaluar = st.button("Evaluar Proveedor", type="primary",
                            use_container_width=True,
                            disabled=not nombre_proveedor)
    with col_b2:
        if st.button("Limpiar formulario", use_container_width=True):
            st.session_state.form_count += 1
            st.rerun()

    # PREDICCION
    if evaluar and nombre_proveedor:
        mediana_global  = config.get('mediana_global_delivery', 12.0)
        defect_rate     = unidades_defectuosas / cantidad if cantidad > 0 else 0
        savings_pct     = (precio_unitario - precio_negociado) / precio_unitario if precio_unitario > 0 else 0
        cancelled_flag  = 1 if cancelada else 0
        delay_vs_median = (dias_entrega - mediana_global) if not cancelada else 0

        input_data = pd.DataFrame([{
            'defect_rate'    : defect_rate,
            'savings_pct'    : savings_pct,
            'cancelled_flag' : cancelled_flag,
            'delay_vs_median': delay_vs_median
        }])[FEATURES]

        pred_proba = model.predict(scaler.transform(input_data), verbose=0)[0]
        pred_idx   = int(np.argmax(pred_proba))
        pred_clase = LABEL_MAP_INV[pred_idx]
        color      = COLORES[pred_clase]

        st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
        st.subheader(f"Resultado: {nombre_proveedor}")

        st.markdown(f"""
        <div class="riesgo-{pred_clase.lower()}">
            <h2 style="margin:0;color:{color}">Riesgo {pred_clase.upper()}</h2>
            <p style="margin:4px 0 0;color:#a0a0c0">Confianza del modelo: <strong>{pred_proba[pred_idx]:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Bajo",  f"{pred_proba[0]:.1%}")
        col_p2.metric("Medio", f"{pred_proba[1]:.1%}")
        col_p3.metric("Alto",  f"{pred_proba[2]:.1%}")

        st.markdown("**Recomendacion:**")
        if pred_clase == 'Alto':
            st.error("Accion inmediata: Revisar contrato y SLA | Solicitar plan de mejora en 15 dias | Evaluar proveedor alternativo | Incrementar inspecciones de calidad.")
        elif pred_clase == 'Medio':
            st.warning("Seguimiento: Programar revision de desempeno | Monitorear proximas 3 entregas | Negociar mejora en tiempos o defectos.")
        else:
            st.success("Proveedor confiable: Mantener relacion comercial | Considerar para mayor volumen | Incluir en programa de preferentes.")

        st.session_state.evaluaciones.append({
            'Empresa'         : st.session_state.get('empresa', 'Sin definir'),
            'Area'            : st.session_state.get('area',    'Sin definir'),
            'Proveedor'       : nombre_proveedor,
            'Categoria'       : categoria,
            'Cantidad'        : cantidad,
            'Precio unitario' : precio_unitario,
            'Precio negociado': precio_negociado,
            'Unidades defect.': unidades_defectuosas,
            'Dias entrega'    : dias_entrega if not cancelada else 'Cancelada',
            'Cancelada'       : 'Si' if cancelada else 'No',
            'Tasa defectos'   : f"{defect_rate:.1%}",
            'Ahorro negociado': f"{savings_pct:.1%}",
            'Riesgo'          : pred_clase,
            'Confianza'       : f"{pred_proba[pred_idx]:.1%}",
            'Prob. Bajo'      : f"{pred_proba[0]:.1%}",
            'Prob. Medio'     : f"{pred_proba[1]:.1%}",
            'Prob. Alto'      : f"{pred_proba[2]:.1%}",
            'Fecha evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M')
        })
        st.success(f"Evaluacion guardada. Total en sesion: {len(st.session_state.evaluaciones)}")

# ============================================================
# TAB: HISTORIAL
# ============================================================
with tab2:
    st.subheader("Historial de Evaluaciones")

    if not st.session_state.evaluaciones:
        st.info("Aun no hay evaluaciones. Usa el formulario para agregar proveedores.")
    else:
        df_hist = pd.DataFrame(st.session_state.evaluaciones)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Total",        len(df_hist))
        col_r2.metric("Riesgo Alto",  (df_hist['Riesgo'] == 'Alto').sum())
        col_r3.metric("Riesgo Medio", (df_hist['Riesgo'] == 'Medio').sum())
        col_r4.metric("Riesgo Bajo",  (df_hist['Riesgo'] == 'Bajo').sum())

        st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)

        def color_riesgo(val):
            mapa = {'Alto': '#3d1515', 'Medio': '#3d3415', 'Bajo': '#153d1e'}
            return f"background-color: {mapa.get(val, '')}"

        cols_mostrar = ['Proveedor','Categoria','Tasa defectos','Dias entrega',
                        'Cancelada','Riesgo','Confianza','Fecha evaluacion']
        st.dataframe(
            df_hist[cols_mostrar].style.applymap(color_riesgo, subset=['Riesgo']),
            use_container_width=True, hide_index=True
        )

        st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
        st.subheader("Exportar")

        fecha_hoy      = datetime.now().strftime('%Y%m%d_%H%M')
        nombre_empresa = (st.session_state.get('empresa') or 'SmartSupplier').replace(' ', '_')

        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_hist.to_excel(writer, index=False, sheet_name='Evaluaciones')
            buffer.seek(0)
            st.download_button("Descargar Excel", data=buffer,
                file_name=f"SmartSupplier_{nombre_empresa}_{fecha_hoy}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

        with col_d2:
            lineas = ["SmartSupplier AI | GP Intelligence",
                      f"Empresa: {nombre_empresa} | Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      "=" * 60]
            for _, row in df_hist.iterrows():
                lineas += [f"\nProveedor: {row['Proveedor']}",
                           f"  Riesgo   : {row['Riesgo']} ({row['Confianza']})",
                           f"  Defectos : {row['Tasa defectos']}",
                           f"  Entrega  : {row['Dias entrega']} | Cancelada: {row['Cancelada']}",
                           f"  Evaluado : {row['Fecha evaluacion']}"]
            st.download_button("Descargar TXT",
                data="\n".join(lineas).encode('utf-8'),
                file_name=f"SmartSupplier_{nombre_empresa}_{fecha_hoy}.txt",
                mime="text/plain", use_container_width=True)

        with col_d3:
            if st.button("Limpiar historial", use_container_width=True):
                st.session_state.evaluaciones = []
                st.rerun()

        st.info("Envio por correo automatico disponible en n8n (siguiente paso del proyecto).")

# ============================================================
# FOOTER
# ============================================================
st.markdown('<hr class="gp-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer-gp">
    <div class="footer-name">Ines Margarita Gonzalez Ponton</div>
    <div class="footer-role">Ingeniera Industrial | Analista de Procesos | IA y Deep Learning</div>
    <div class="footer-contact">
        Tel: +57 323-2932832 &nbsp;|&nbsp; imgp802@outlook.es &nbsp;|&nbsp;
        linkedin.com/in/ines-margarita-gonzalez-ponton-295962205
    </div>
    <div class="footer-brand">GP Intelligence | Engineering Intelligent Decisions | SmartSupplier AI v1.0</div>
</div>
""", unsafe_allow_html=True)

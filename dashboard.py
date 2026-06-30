import time
import cv2
import streamlit as st

# Importações do seu projeto
from camera.capture import CameraCapture
from vision.renderer import Renderer
from services.analyzer import Analyzer 
from vision.preprocessor import Preprocessor
from vision.contours import ContourDetector

# Importação das suas novas classes de Estoque
from count.counter import ProfileCounter
from count.event_manager import EventManager

# 1. Configuração da Página Web
st.set_page_config(page_title="Sistema de Inspeção | System.AI", layout="wide", initial_sidebar_state="expanded")

# --- 2. INJEÇÃO DE CSS CUSTOMIZADO ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3, h4 { color: #F0F2F6 !important; font-family: 'Segoe UI', sans-serif; }
    hr { border: 0; height: 2px; background: linear-gradient(to right, #C41230, #0033A0); margin: 1rem 0; }
    
    /* Métricas (Azul Senai para Valores) */
    [data-testid="stMetricValue"] { color: #0033A0 !important; font-size: 1.8rem !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #A0AEC0 !important; font-size: 1rem !important; }
    
    /* Métrica Gigante para o Estoque (Vermelho Mecald) */
    .estoque-valor [data-testid="stMetricValue"] { color: #C41230 !important; font-size: 4rem !important; }
    
    div.stButton > button[kind="primary"] { background-color: #C41230; color: white; border: none; width: 100%; font-weight: bold; }
    div.stButton > button[kind="primary"]:hover { background-color: #A00F25; }
    div.stButton > button[kind="secondary"] { background-color: #0033A0; color: white; border: none; width: 100%; font-weight: bold; }
    div.stButton > button[kind="secondary"]:hover { background-color: #002277; }
    </style>
""", unsafe_allow_html=True)

# Logos
logo_mecald = "images/mecald.png"
logo_senai = "images/senai.png"
logo_system_ai = "images/systemai.png"

# --- 3. CABEÇALHO ---
col_titulo, col_logo_top = st.columns([85, 15], gap="large")
with col_titulo:
    st.markdown("<h1>🏭 Centro de Controle | System.AI</h1>", unsafe_allow_html=True)
    st.caption("Inspeção de Qualidade & Gestão de Estoque em Tempo Real")
with col_logo_top:
    st.image(logo_system_ai, width=120)

st.markdown("---")

# --- 4. BARRA LATERAL ---
st.sidebar.image(logo_system_ai, width=150)
st.sidebar.markdown("## ⚙️ Setup de Linha")
materiais_opcoes = {"Aço Carbono": 0, "Aço Inox": 1, "Alumínio": 2}
material_id = materiais_opcoes[st.sidebar.selectbox("Lote Atual (Material):", list(materiais_opcoes.keys()))]

estoque_inicial = st.sidebar.number_input("Estoque Inicial:", min_value=0, value=0, step=1)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2 = st.sidebar.columns(2)
iniciar = col_btn1.button("▶ INICIAR", type="primary")
parar = col_btn2.button("⏹ PARAR", type="secondary")

if "rodando" not in st.session_state:
    st.session_state.rodando = False
    st.session_state.estoque_atual = estoque_inicial
    st.session_state.ultimo_evento = "Nenhum evento registrado."

if iniciar:
    st.session_state.rodando = True
    st.session_state.estoque_atual = estoque_inicial
if parar:
    st.session_state.rodando = False

# --- 5. LAYOUT PRINCIPAL (2 Telas Independentes) ---
col_qualidade, col_estoque = st.columns(2, gap="large")

with col_qualidade:
    st.markdown("### 🔬 CAM 1: Inspeção de Qualidade")
    video_qualidade = st.empty()
    status_qualidade = st.empty()
    barra_progresso = st.empty()
    painel_metricas = st.empty()
    painel_resultado = st.empty()

with col_estoque:
    st.markdown("### 📦 CAM 2: Monitoramento de Estoque")
    video_estoque = st.empty()
    
    st.markdown("---")
    st.markdown("### 📊 Saldo em Estoque")
    
    # Criamos um container com a classe CSS customizada para deixar o número gigante e vermelho
    container_estoque = st.container()
    estoque_metric = container_estoque.empty()
    evento_log = container_estoque.empty()
    
    # Mostra o valor inicial
    estoque_metric.markdown('<div class="estoque-valor">', unsafe_allow_html=True)
    estoque_metric.metric(label="Quantidade de Perfilados", value=st.session_state.estoque_atual)
    estoque_metric.markdown('</div>', unsafe_allow_html=True)

st.write("\n" * 3)
col_espaco, col_rodape = st.columns([75, 25])
with col_rodape:
    col_l1, col_l2 = st.columns(2)
    with col_l1: st.image(logo_mecald, width=110)
    with col_l2: st.image(logo_senai, width=100)


# --- 6. MOTOR DO SISTEMA MULTI-CÂMERA ---
if st.session_state.rodando:
    # Instâncias CAM 1 (Qualidade)
    cam_qualidade = CameraCapture(source=0) # Adapte o parâmetro conforme sua classe
    renderer = Renderer()
    analyzer = Analyzer(samples_target=60)
    
    # Instâncias CAM 2 (Estoque)
    cam_estoque = CameraCapture(source=1) # Adapte para a segunda câmera
    preprocessor_estoque = Preprocessor()
    detector_estoque = ContourDetector()
    counter = ProfileCounter()
    event_manager = EventManager(confirmation_time=3)
    
    # FIX: Definindo o estoque inicial para a máquina de estados não travar
    event_manager.confirmed_count = st.session_state.estoque_atual
    
    cam_qualidade.connect()
    cam_estoque.connect()
    start_time = time.time()
    capture_delay = 10 
    samples_count = 0

    painel_resultado.info("⏳ Aguardando posicionamento da peça para inspeção...")
    evento_log.caption(f"Último log: {st.session_state.ultimo_evento}")

    while st.session_state.rodando:
        # Lendo ambas as câmeras
        frame_q = cam_qualidade.read_frame()
        frame_e = cam_estoque.read_frame()

        if frame_q is None and frame_e is None:
            time.sleep(0.05)
            continue

        # ==========================================
        # ROTINA DA CÂMERA 2 (ESTOQUE)
        # ==========================================
        if frame_e is not None:
            frame_e_exibicao = frame_e.copy()
            
            # Processamento visual básico para alimentar o counter
            processed_e = preprocessor_estoque.process(frame_e)
            _, _, all_contours = detector_estoque.find_largest(processed_e)
            
            # Garantir que passamos uma lista vazia se for None para o validate_input não falhar
            lista_contornos = all_contours if all_contours is not None else []
            
            # Executando a lógica de contagem e eventos
            current_count = counter.count(lista_contornos)
            evento = event_manager.update(current_count)
            
            # Desenha o número de peças detectadas no frame para o operador ver
            cv2.putText(frame_e_exibicao, f"Detectados Agora: {current_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Se a máquina de estados confirmou uma mudança (ENTRY ou EXIT)
            if evento:
                st.session_state.estoque_atual = evento["current_count"]
                st.session_state.ultimo_evento = f"[{evento['timestamp']}] {evento['event_type']}: Variação de {evento['difference']} peça(s)"
                
                # Atualiza a tela em tempo real
                estoque_metric.markdown('<div class="estoque-valor">', unsafe_allow_html=True)
                estoque_metric.metric(label="Quantidade de Perfilados", value=st.session_state.estoque_atual, delta=evento['difference'] if evento['event_type'] == 'ENTRY' else -evento['difference'])
                estoque_metric.markdown('</div>', unsafe_allow_html=True)
                evento_log.caption(f"Último log: {st.session_state.ultimo_evento}")
                
            video_estoque.image(frame_e_exibicao, channels="BGR", use_column_width=True)

        # ==========================================
        # ROTINA DA CÂMERA 1 (QUALIDADE)
        # ==========================================
        if frame_q is not None:
            elapsed = time.time() - start_time
            frame_q_exibicao = frame_q.copy()

            if elapsed < capture_delay:
                remaining = capture_delay - elapsed
                cv2.putText(frame_q_exibicao, f"Calibrando... ({remaining:.1f}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                video_qualidade.image(frame_q_exibicao, channels="BGR", use_column_width=True)
                status_qualidade.warning(f"**Atenção:** Mantenha a peça estática.")
                continue

            result = analyzer.analyze(frame_q, material_id)
            features = result.get("features")

            if features is not None:
                renderer.draw_complete_overlay(frame_q_exibicao, features)
                
                if not result["ready"]:
                    samples_count += 1
                    barra_progresso.progress(min(samples_count / 60.0, 1.0))
                    status_qualidade.info(f"**Capturando Frames:** {samples_count}/60 amostras")
                    
                    with painel_metricas.container():
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric(label="Circularidade", value=f"{features['circularity']:.3f}")
                        m_col2.metric(label="Aspect Ratio", value=f"{features['aspect_ratio']:.3f}")

            if result["ready"]:
                barra_progresso.progress(1.0)
                status_qualidade.success("✅ **Leitura concluída.**")
                
                if result["is_confiable"]:
                    painel_resultado.success(f"### ⚙️ Peça: {result['prediction']}\n**Confiança:** {result['confidence'] * 100:.1f}%")
                else:
                    painel_resultado.error(f"### ⚠️ {result['prediction']}\n**Confiança:** {result['confidence'] * 100:.1f}%")
                
                st.session_state.rodando = False
                break

            video_qualidade.image(frame_q_exibicao, channels="BGR", use_column_width=True)

        time.sleep(0.01)

    cam_qualidade.release()
    cam_estoque.release()
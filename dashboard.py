import time

import cv2
import streamlit as st

# Importações do seu projeto
from camera.capture import CameraCapture
from services.analyzer import Analyzer
from vision.renderer import Renderer

# 1. Configuração da Página Web (Deve ser a primeira linha do Streamlit)
st.set_page_config(
    page_title="Sistema de Inspeção | System.AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. INJEÇÃO DE CSS CUSTOMIZADO (Design System) ---
# Aqui definimos as cores: Fundo Escuro, Vermelho Mecald e Azul Senai
st.markdown(
    """
    <style>
    /* Cor de fundo principal (Preto/Cinza Escuro Industrial) */
    .stApp {
        background-color: #0E1117;
    }

    /* Customização do Cabeçalho e Títulos */
    h1, h2, h3 {
        color: #F0F2F6 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Linhas divisórias com gradiente Vermelho -> Azul */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(to right, #C41230, #0033A0);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Customização dos painéis de Métrica (Dashboard) */
    [data-testid="stMetricValue"] {
        color: #0033A0 !important; /* Azul Senai para os números */
        font-size: 1.8rem !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #A0AEC0 !important;
        font-size: 1rem !important;
    }

    /* Botão Primário (Iniciar) - Vermelho Mecald/System.AI */
    div.stButton > button[kind="primary"] {
        background-color: #C41230;
        color: white;
        border: none;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #A00F25;
        border: none;
    }

    /* Botão Secundário (Parar) - Azul Senai */
    div.stButton > button[kind="secondary"] {
        background-color: #0033A0;
        color: white;
        border: none;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #002277;
        border: none;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Definição dos arquivos das logos
logo_mecald = "images/mecald.png"
logo_senai = "images/senai.png"
logo_system_ai = "images/systemai.png"

# --- 3. CABEÇALHO ---
col_titulo, col_logo_top = st.columns([85, 15], gap="large")
with col_titulo:
    st.markdown("<h1>🏭 Inspeção Automática de Perfilados</h1>", unsafe_allow_html=True)
    st.caption(
        "Visão Computacional e Inteligência Artificial para Controle de Qualidade"
    )
with col_logo_top:
    st.image(logo_system_ai, width=120)

st.markdown("---")  # Linha divisória com o gradiente das empresas

# --- 4. BARRA LATERAL (Controles) ---
st.sidebar.image(logo_system_ai, width=150)
st.sidebar.markdown("## ⚙️ Setup de Linha")

materiais_opcoes = {"Aço Carbono": 0, "Aço Inox": 1, "Alumínio": 2}
material_selecionado = st.sidebar.selectbox(
    "Lote Atual (Material):", list(materiais_opcoes.keys())
)
material_id = materiais_opcoes[material_selecionado]

st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Espaçamento

col_btn1, col_btn2 = st.sidebar.columns(2)
iniciar = col_btn1.button("▶ INICIAR", type="primary")  # Fica vermelho
parar = col_btn2.button("⏹ PARAR", type="secondary")  # Fica azul

if "rodando" not in st.session_state:
    st.session_state.rodando = False

if iniciar:
    st.session_state.rodando = True
if parar:
    st.session_state.rodando = False

# --- 5. LAYOUT PRINCIPAL DO DASHBOARD ---
# 60% para câmeras, 40% para dados analíticos
coluna_video, coluna_dados = st.columns([6, 4], gap="large")

with coluna_video:
    st.markdown("### 📷 Monitoramento em Tempo Real")
    video_placeholder = st.empty()

    st.markdown("### 👁️ Visão da Máquina (Threshold)")
    thresh_placeholder = st.empty()

with coluna_dados:
    st.markdown("### 📊 Análise da Peça")
    status_texto = st.empty()
    barra_progresso = st.empty()

    st.markdown("---")
    st.markdown("### 📐 Características Físicas (Features)")

    # Criando o Dashboard de Métricas (Muito mais limpo que texto solto)
    painel_metricas = st.empty()

    st.markdown("---")
    st.markdown("### 🤖 Diagnóstico do Modelo")
    painel_resultado = st.empty()

# --- 6. RODAPÉ ---
st.write("\n" * 5)  # Espaçamento dinâmico
col_espaco, col_rodape = st.columns([75, 25])
with col_rodape:
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.image(logo_mecald, width=110)
    with col_l2:
        st.image(logo_senai, width=100)


# --- 7. MOTOR DO SISTEMA (Loop do OpenCV) ---
if st.session_state.rodando:
    camera = CameraCapture()
    renderer = Renderer()
    analyzer = Analyzer(samples_target=60)

    camera.connect()
    start_time = time.time()
    capture_delay = 15
    samples_count = 0

    painel_resultado.info("⏳ Aguardando posicionamento da peça na esteira...")

    while st.session_state.rodando:
        frame = camera.read_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        elapsed = time.time() - start_time
        frame_exibicao = frame.copy()

        # Fase 1: Setup Inicial
        if elapsed < capture_delay:
            remaining = capture_delay - elapsed
            cv2.putText(
                frame_exibicao,
                f"Calibrando... ({remaining:.1f}s)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
            )
            video_placeholder.image(
                frame_exibicao, channels="BGR", use_column_width=True
            )
            status_texto.warning(
                f"**Atenção:** Mantenha a peça estática. Iniciando leitura em {remaining:.1f}s"
            )
            continue

        # Fase 2: Extração de Dados
        result = analyzer.analyze(frame, material_id)
        features = result.get("features")
        processed_frame = result.get("processed_frame")

        if features is not None:
            renderer.draw_complete_overlay(frame_exibicao, features)

            if not result["ready"]:
                samples_count += 1
                progresso_atual = min(samples_count / 60.0, 1.0)
                barra_progresso.progress(progresso_atual)
                status_texto.info(f"**Capturando Frames:** {samples_count}/60 amostras")

                # Renderizando as features como cartões de métricas (Visual Industrial)
                with painel_metricas.container():
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric(
                        label="Circularidade", value=f"{features['circularity']:.3f}"
                    )
                    m_col2.metric(
                        label="Aspect Ratio", value=f"{features['aspect_ratio']:.3f}"
                    )

                    m_col3, m_col4 = st.columns(2)
                    m_col3.metric(label="Furos Internos", value=f"{features['holes']}")
                    vazado_texto = "Sim" if features["is_hollow"] else "Não"
                    m_col4.metric(label="Perfil Vazado?", value=vazado_texto)

        # Fase 3: Conclusão e Resultado
        if result["ready"]:
            barra_progresso.progress(1.0)
            status_texto.success("✅ **Leitura concluída com sucesso.**")

            if result["is_confiable"]:
                painel_resultado.success(
                    f"### ⚙️ Peça: {result['prediction']}\n**Confiança da IA:** {result['confidence'] * 100:.1f}%"
                )
            else:
                painel_resultado.error(
                    f"### ⚠️ {result['prediction']}\n**Confiança da IA:** {result['confidence'] * 100:.1f}%\n*Requer verificação manual.*"
                )

            st.session_state.rodando = False
            break

        # Atualização Visual
        video_placeholder.image(frame_exibicao, channels="BGR", use_column_width=True)
        if processed_frame is not None:
            thresh_placeholder.image(processed_frame, use_column_width=True)

        time.sleep(0.01)

    camera.release()

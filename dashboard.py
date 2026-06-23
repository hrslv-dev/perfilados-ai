import time

import cv2
import streamlit as st

# Importações do seu projeto (ajuste os caminhos se necessário)
from camera.capture import CameraCapture
from services.analyzer import (
    Analyzer,  # Ajuste para a pasta correta onde está o Analyzer
)
from vision.renderer import Renderer

# 1. Configuração da Página Web
st.set_page_config(page_title="Inspeção de Perfilados", layout="wide")
st.title("🏭 Sistema de Inspeção de Perfilados Industriais")

# 2. Barra Lateral (Menu de Configurações)
st.sidebar.header("Configurações")
materiais_opcoes = {"Aço Carbono": 0, "Aço Inox": 1, "Alumínio": 2}
material_selecionado = st.sidebar.selectbox(
    "Selecione o Material da Linha:", list(materiais_opcoes.keys())
)
material_id = materiais_opcoes[material_selecionado]

# Botões de Controle
col_btn1, col_btn2 = st.sidebar.columns(2)
iniciar = col_btn1.button("▶ Iniciar Sistema", type="primary")
parar = col_btn2.button("⏹ Parar")

# Controle de estado para manter o loop rodando
if "rodando" not in st.session_state:
    st.session_state.rodando = False

if iniciar:
    st.session_state.rodando = True
if parar:
    st.session_state.rodando = False

# 3. Layout Principal (Dividido em duas colunas)
coluna_video, coluna_dados = st.columns([6, 4])

with coluna_video:
    st.subheader("Câmera ao Vivo")
    video_placeholder = st.empty()  # Aqui vai o frame principal

    st.subheader("Processamento (Threshold)")
    thresh_placeholder = st.empty()  # Aqui vai o frame preto e branco

with coluna_dados:
    st.subheader("Status da Análise")
    barra_progresso = st.empty()
    status_texto = st.empty()

    st.markdown("---")
    st.subheader("Features em Tempo Real")
    painel_features = st.empty()

    st.markdown("---")
    st.subheader("Resultado da Predição")
    painel_resultado = st.empty()

# 4. O Loop Principal do Sistema
if st.session_state.rodando:
    # Inicializa as classes
    camera = CameraCapture()
    renderer = Renderer()
    analyzer = Analyzer(samples_target=60)

    camera.connect()
    start_time = time.time()
    capture_delay = 15  # segundos
    samples_count = 0

    # Limpa painéis de resultado de testes anteriores
    painel_resultado.info("Aguardando coleta de amostras...")

    while st.session_state.rodando:
        frame = camera.read_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        elapsed = time.time() - start_time
        frame_exibicao = frame.copy()  # Cópia para desenhar sem alterar o original

        # Fase 1: Aguardando delay inicial
        if elapsed < capture_delay:
            remaining = capture_delay - elapsed
            cv2.putText(
                frame_exibicao,
                f"Iniciando em... ({remaining:.1f}s)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
            )
            video_placeholder.image(frame_exibicao, channels="BGR")
            status_texto.warning(f"Posicione a peça. Iniciando em {remaining:.1f}s")
            continue

        # Fase 2: Análise e Coleta
        result = analyzer.analyze(frame, material_id)
        features = result.get("features")
        processed_frame = result.get("processed_frame")

        if features is not None:
            # Desenha as marcações na tela
            renderer.draw_complete_overlay(frame_exibicao, features)

            if not result["ready"]:
                samples_count += 1
                # Atualiza a barra de progresso (limita a 1.0 = 100%)
                progresso_atual = min(samples_count / 60.0, 1.0)
                barra_progresso.progress(progresso_atual)
                status_texto.text(f"Coletando amostras: {samples_count}/60")

                # Exibe as features como um dashboard
                painel_features.markdown(f"""
                * **Circularidade:** `{features["circularity"]:.3f}`
                * **Aspect Ratio:** `{features["aspect_ratio"]:.3f}`
                * **Furos Identificados:** `{features["holes"]}`
                * **É vazado?** `{"Sim" if features["is_hollow"] else "Não"}`
                * **Área do Contorno:** `{features["area"]:.0f} px`
                """)

        # Fase 3: Exibe o Resultado da Predição
        if result["ready"]:
            barra_progresso.progress(1.0)
            status_texto.success("Coleta Concluída!")

            # Formata a mensagem de resultado baseada na confiança
            if result["is_confiable"]:
                painel_resultado.success(
                    f"✅ **Peça Identificada:** {result['prediction']}"
                )
            else:
                painel_resultado.error(f"⚠️ **Atenção:** {result['prediction']}")

            painel_resultado.markdown(f"""
            **Nível de Confiança:** `{result["confidence"] * 100:.2f}%`
            *Detalhes do Modelo:* {result["message"]}
            """)

            # Para o loop após classificar a peça (para a tela congelar no resultado)
            st.session_state.rodando = False
            break

        # Atualiza as imagens na interface web (converte para o padrão Streamlit)
        video_placeholder.image(frame_exibicao, channels="BGR", use_column_width=True)
        if processed_frame is not None:
            thresh_placeholder.image(processed_frame, use_column_width=True)

        # Pequena pausa para não travar a interface web
        time.sleep(0.01)

    camera.release()

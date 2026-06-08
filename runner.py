"""
capturar_perfil.py
════════════════════════════════════════════════════════════════════════════════
Unifica os dois mundos do projeto: visão computacional + modelo ML.

FLUXO EM 4 FASES:
  ① POSICIONAMENTO → Countdown de X segundos para o operador posicionar o perfil
  ② CAPTURA        → N frames capturados, features extraídas de cada um
  ③ AGREGAÇÃO      → Média estatística de todas as features válidas
  ④ PREDIÇÃO       → Dict agregado enviado ao RandomForest → resultado + confiança

POR QUE TIRAR A MÉDIA DE N FRAMES EM VEZ DE UM SÓ?
────────────────────────────────────────────────────────────────────────────────
Uma única captura é ruidosa:
  • O threshold adaptativo pode oscilar entre frames (variação de iluminação)
  • O contorno pode ter pixelização diferente a cada frame
  • Pequenas vibrações mecânicas afetam os valores numéricos das features

Ao capturar N frames e calcular a média:
  • O ruído aleatório se cancela (Lei dos Grandes Números)
  • A feature "real" do objeto emerge com muito mais confiabilidade
  • O desvio padrão entre frames mede a ESTABILIDADE da leitura

COMO EXECUTAR:
────────────────────────────────────────────────────────────────────────────────
  # Uso básico — prompts interativos para material
  python capturar_perfil.py

  # Especificando material via argumento
  python capturar_perfil.py --material aco_carbono

  # Configurando todos os parâmetros
  python capturar_perfil.py --material inox --countdown 8 --frames 80 --camera 0

ARGUMENTOS DISPONÍVEIS:
  --countdown   N de segundos para posicionar o perfil (padrão: 5)
  --frames      N de frames a capturar (padrão: 60)
  --material    Material do perfil: aco_carbono | inox | aluminio (opcional)
  --camera      Índice da câmera (padrão: 0)
  --min-area    Área mínima do contorno em px² (padrão: 2000)
  --model-dir   Pasta do modelo treinado (padrão: ml/)

DEPENDÊNCIAS:
  • vision/preprocessor.py    → CLAHE + threshold adaptativo
  • vision/contours.py        → detecção do contorno principal
  • vision/features.py        → extração das features numéricas
  • vision/renderer.py        → overlay visual na janela cv2
  • camera/capture.py         → interface com a câmera
  • ml/model.py               → Classifier (RandomForest)
  • ml/random_forest.pkl      → modelo treinado (gerado por training.py)
"""

# ─── Imports da biblioteca padrão ─────────────────────────────────────────────
import sys
import time
import argparse
from pathlib import Path

# ─── Imports de terceiros ──────────────────────────────────────────────────────
import cv2
import numpy as np

# ─── Imports do projeto ────────────────────────────────────────────────────────
# Todos os módulos abaixo estão na mesma estrutura de pastas do test.py.
# Se este arquivo estiver na raiz do projeto, os imports funcionam igualmente.

from camera.capture      import CameraCapture
from vision.preprocessor import Preprocessor
from vision.contours     import ContourDetector
from vision.features     import FeatureExtractor
from vision.renderer     import Renderer


# ════════════════════════════════════════════════════════════════════════════════
#  MAPEAMENTO DE MATERIAIS
#  Traduz string legível → código numérico que o modelo entende.
# ════════════════════════════════════════════════════════════════════════════════

MATERIAL_MAP = {
    "aco_carbono": 0,
    "inox":        1,
    "aluminio":    2,
}

MATERIAL_LABEL = {v: k for k, v in MATERIAL_MAP.items()}

# ════════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES DE INTERFACE VISUAL NO TERMINAL
#  Separam a lógica de impressão do código principal — mais fácil de manter.
# ════════════════════════════════════════════════════════════════════════════════

def linha(char="═", n=65):
    """Imprime uma linha separadora."""
    print(char * n)

def titulo(texto, char="═"):
    """Imprime um título centralizado com separadores."""
    print()
    linha(char)
    print(f"  {texto}")
    linha(char)


def _barra_progresso(atual, total, largura=30):
    """
    Gera uma string de barra de progresso.
    Ex: [████████████░░░░░░░░░░░░░░░░░░] 40%
    """
    preenchido = int(largura * atual / total)
    barra = "█" * preenchido + "░" * (largura - preenchido)
    pct   = int(atual / total * 100)
    return f"[{barra}] {pct:3d}%"


# ════════════════════════════════════════════════════════════════════════════════
#  OVERLAY NA JANELA DO cv2
#  Funções que desenham texto e elementos visuais no frame da câmera.
# ════════════════════════════════════════════════════════════════════════════════

def desenhar_countdown_frame(frame, segundos_restantes):
    """
    Desenha o countdown sobre o frame da câmera.
    O operador vê em tempo real quanto tempo tem para posicionar o perfil.
    """
    h, w = frame.shape[:2]

    # Fundo semitransparente para legibilidade
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Número grande do countdown
    cv2.putText(
        frame,
        str(segundos_restantes),
        (w // 2 - 20, h // 2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        5.0,                      # fonte muito grande para visibilidade
        (0, 220, 255),            # amarelo
        8,
        cv2.LINE_AA
    )

    # Instrução na parte inferior
    cv2.putText(
        frame,
        "Posicione o perfil na camera...",
        (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2
    )


def desenhar_captura_frame(frame, frame_idx, total_frames, features_atuais):
    """
    Desenha informações de progresso e features durante a captura.
    O operador vê que a captura está acontecendo e as features sendo medidas.
    """
    h, w = frame.shape[:2]

    # Fundo semitransparente na parte superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Barra de progresso visual no frame
    progresso = int((frame_idx / total_frames) * w)
    cv2.rectangle(frame, (0, 105), (progresso, 110), (0, 200, 100), -1)
    cv2.rectangle(frame, (0, 105), (w, 110), (80, 80, 80), 1)

    # Texto: progresso
    cv2.putText(
        frame,
        f"Capturando: {frame_idx}/{total_frames}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 220, 100),
        2
    )

    # Features em tempo real (se contorno detectado)
    if features_atuais:
        circ = features_atuais.get("circularity", 0)
        ar   = features_atuais.get("aspect_ratio", 0)
        area = features_atuais.get("area", 0)
        hl   = features_atuais.get("holes", 0)

        cv2.putText(
            frame,
            f"circ={circ:.3f}  ar={ar:.3f}  area={area:.0f}  holes={hl}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1
        )
    else:
        cv2.putText(
            frame,
            "Sem contorno — aguardando perfil...",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 255),
            1
        )


def desenhar_resultado_frame(frame, resultado):
    """
    Exibe o resultado final da predição sobre o frame.
    Chamado após a fase de agregação.
    """
    h, w = frame.shape[:2]

    label      = resultado.get("label", "erro")
    conf       = resultado.get("confidence_pct", 0)
    baixa_conf = resultado.get("low_confidence", True)

    cor = (0, 200, 100) if not baixa_conf else (0, 165, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame,
        label.replace("_", " ").title(),
        (15, h - 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        cor,
        2
    )
    cv2.putText(
        frame,
        f"Confianca: {conf}%",
        (15, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        cor,
        2
    )
    if baixa_conf:
        cv2.putText(
            frame,
            "CONFIRME VISUALMENTE",
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 100, 255),
            2
        )


# ════════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPAL: CapturaEClassificador
#  Orquestra o pipeline completo em 4 fases.
# ════════════════════════════════════════════════════════════════════════════════

class CapturaEClassificador:
    """
    Orquestra o pipeline completo:
      Câmera → Pré-processamento → Extração de Features →
      Agregação Estatística → Predição ML → Resultado

    Por que separar em uma classe?
    ────────────────────────────────────────────────────────────────────────────
    Manter estado interno entre as fases (frames coletados, médias acumuladas,
    câmera aberta) é mais limpo em uma classe do que com variáveis globais.
    Facilita também adicionar novas fases ou modificar o comportamento.
    """

    def __init__(
        self,
        n_frames    : int  = 60,
        countdown   : int  = 5,
        camera_id   : int  = 0,
        material    : int  = 0,
        min_area    : int  = 2000,
        model_dir   : str  = "ml/",
        min_validos : float = 0.30,   # fração mínima de frames com contorno
    ):
        """
        Inicializa os componentes do pipeline.

        Parâmetros:
            n_frames    : quantos frames serão capturados na fase de coleta
            countdown   : segundos de espera antes de iniciar a captura
            camera_id   : índice da câmera (0 = câmera padrão)
            material    : código do material (0=carbono, 1=inox, 2=alumínio)
            min_area    : área mínima do contorno em px² — filtra ruído
            model_dir   : diretório onde estão os arquivos .pkl e .json
            min_validos : fração mínima de frames válidos para aceitar resultado
        """
        self.n_frames    = n_frames
        self.countdown   = countdown
        self.material    = material
        self.min_validos = min_validos

        # ── Instancia módulos de visão ────────────────────────────────────────
        self.camera      = CameraCapture(source=camera_id)
        self.preprocessor = Preprocessor()
        self.detector    = ContourDetector(min_area=min_area)
        self.extractor   = FeatureExtractor()
        self.renderer    = Renderer()

        # ── Lista interna onde armazenamos features de cada frame válido ──────
        # Será preenchida na fase de captura e consumida na fase de agregação.
        self._features_por_frame: list[dict] = []

        # ── Carrega o modelo ML ───────────────────────────────────────────────
        # Importamos aqui (não no topo do arquivo) para o script funcionar
        # mesmo sem o modelo treinado — exibindo mensagem amigável de erro.
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from ml.model import Classifier
            self.classificador = Classifier()
        except FileNotFoundError as e:
            print(f"\n[AVISO] Modelo não encontrado: {e}")
            print("  Execute: python ml/training.py")
            print("  O script continuará, mas a predição será pulada.\n")
            self.classificador = None

    # ──────────────────────────────────────────────────────────────────────────
    #  FASE 1 — COUNTDOWN
    # ──────────────────────────────────────────────────────────────────────────

    def fase_countdown(self):
        """
        Exibe o countdown enquanto mantém a janela da câmera aberta.
        O operador usa esse tempo para posicionar o perfil corretamente.

        POR QUE usar cv2.waitKey(100) em vez de time.sleep(1)?
        ────────────────────────────────────────────────────────────────────────
        cv2.waitKey(100) espera 100ms E processa eventos da janela OpenCV.
        Se usarmos time.sleep(1) sem chamar waitKey, a janela congela
        (não atualiza o frame na tela) porque o loop de eventos do OpenCV
        não é executado. Loopando 10× com waitKey(100) = 1 segundo total.
        """
        titulo("FASE 1 — POSICIONAMENTO DO PERFIL")
        print(f"  Você tem {self.countdown} segundo(s) para posicionar o perfil.")
        print(f"  Material selecionado: {MATERIAL_LABEL.get(self.material, '?')} (código {self.material})")
        print()

        for seg in range(self.countdown, 0, -1):
            print(f"  [ {seg} ] Posicione o perfil em frente à câmera...")

            # Cada segundo = 10 iterações de 100ms
            # Isso mantém a janela responsiva enquanto "esperamos" 1 segundo
            for _ in range(10):
                ret, frame = self.camera.cap.read()
                if not ret:
                    continue

                desenhar_countdown_frame(frame.copy() if frame is not None else frame, seg)

                cv2.imshow("Captura de Perfil — Posicionamento", frame)

                # ESC durante countdown → aborta
                if cv2.waitKey(100) & 0xFF == 27:
                    print("\n  [ABORTADO] ESC pressionado.")
                    return False

        print()
        print("  ✓ Iniciando captura!")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    #  FASE 2 — CAPTURA E EXTRAÇÃO DE FEATURES
    # ──────────────────────────────────────────────────────────────────────────

    def fase_captura(self) -> int:
        """
        Captura N frames e extrai features de cada um.

        Para cada frame:
          1. Lê frame bruto da câmera
          2. Aplica pré-processamento (CLAHE + threshold adaptativo)
          3. Detecta o MAIOR contorno (find_largest — 1 objeto por frame)
          4. Extrai features numéricas do contorno
          5. Armazena no buffer interno self._features_por_frame

        Por que find_largest em vez de find_contours?
        ────────────────────────────────────────────────────────────────────────
        No test.py iteramos sobre TODOS os contornos para validação visual.
        Aqui queremos classificar UM perfil específico — o principal objeto
        da cena. find_largest retorna apenas o contorno de maior área,
        que deve ser o perfil posicionado pelo operador.

        Retorna:
            int : número de frames válidos (com contorno detectado)
        """
        titulo("FASE 2 — CAPTURA E EXTRAÇÃO DE FEATURES")
        print(f"  Capturando {self.n_frames} frames...")
        print(f"  Pressione ESC para abortar.\n")

        frames_validos   = 0
        frames_invalidos = 0
        features_atual   = None   # features do último frame válido (para overlay)

        for idx in range(1, self.n_frames + 1):

            # ── Lê o frame ────────────────────────────────────────────────────
            frame = self.camera.read_frame()

            # ── Pré-processamento ──────────────────────────────────────────────
            # process() retorna imagem binária (0 ou 255 por pixel)
            # pronta para o findContours
            binario = self.preprocessor.process(frame)

            # ── Detecção do contorno principal ─────────────────────────────────
            # find_largest:
            #   1. Filtra contornos com área < min_area (elimina ruído)
            #   2. Retorna o maior contorno válido + hierarquia + todos os contornos
            #
            # Por que receber todos os contornos?
            # → count_holes usa a hierarquia COMPLETA (índices originais).
            #   Se passarmos só o contorno filtrado, os índices não batem.
            contorno, hierarquia, todos_contornos = self.detector.find_largest(binario)

            # ── Frame inválido: nenhum contorno detectado ──────────────────────
            if contorno is None:
                frames_invalidos += 1
                print(
                    f"  [{idx:03d}/{self.n_frames}] "
                    f"✗ sem contorno detectado — frame ignorado "
                    f"(total ignorados: {frames_invalidos})"
                )
                # Atualiza janela mesmo sem contorno (mostra "aguardando")
                desenhar_captura_frame(frame, idx, self.n_frames, None)
                cv2.imshow("Captura de Perfil — Coletando", frame)
                cv2.imshow("Limiarização (CLAHE + Adaptativo)", binario)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # ── Frame válido: extrai features ─────────────────────────────────
            #
            # extract_features recebe:
            #   contorno       : o maior contorno detectado
            #   hierarquia     : hierarquia COMPLETA (necessária para count_holes)
            #   all_contours   : lista COMPLETA de contornos (índices batem com hierarquia)
            #
            # Retorna um dict com: x, y, area, perimeter, width, height,
            #                      aspect_ratio, circularity, holes, is_hollow
            features_frame = self.extractor.extract_features(
                contorno,
                hierarquia,
                all_contours=todos_contornos
            )

            # Adiciona o material ao dict de features deste frame
            # O material não vem do OpenCV — é informação do operador
            features_frame["material"] = self.material

            # Armazena no buffer para agregação posterior
            self._features_por_frame.append(features_frame)
            features_atual = features_frame
            frames_validos += 1

            # ── Print do frame ─────────────────────────────────────────────────
            # Mostra os valores em tempo real para o operador acompanhar.
            # Se os valores estiverem muito instáveis → ambiente não controlado.
            print(
                f"  [{idx:03d}/{self.n_frames}] ✓  "
                f"circ={features_frame['circularity']:.4f} | "
                f"ar={features_frame['aspect_ratio']:.4f} | "
                f"holes={features_frame['holes']} | "
                f"area={features_frame['area']:.1f} | "
                f"hollow={features_frame['is_hollow']}"
            )

            # ── Overlay visual no frame ────────────────────────────────────────
            # Desenha o contorno detectado em verde
            cv2.drawContours(frame, [contorno], -1, (0, 220, 80), 2)
            # Bounding box em azul
            x, y, w, h = (
                features_frame["x"], features_frame["y"],
                features_frame["width"], features_frame["height"]
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 1)

            # Sobreposição de progresso e features no frame
            desenhar_captura_frame(frame, idx, self.n_frames, features_frame)

            cv2.imshow("Captura de Perfil — Coletando", frame)
            cv2.imshow("Limiarização (CLAHE + Adaptativo)", binario)

            if cv2.waitKey(1) & 0xFF == 27:
                print("\n  [ABORTADO] ESC pressionado.")
                break

        print()
        print(f"  ─── Resumo da captura ───────────────────────────────")
        print(f"  Frames totais    : {self.n_frames}")
        print(f"  Frames válidos   : {frames_validos}  ({frames_validos/self.n_frames*100:.1f}%)")
        print(f"  Frames ignorados : {frames_invalidos}  ({frames_invalidos/self.n_frames*100:.1f}%)")

        return frames_validos

    # ──────────────────────────────────────────────────────────────────────────
    #  FASE 3 — AGREGAÇÃO ESTATÍSTICA
    # ──────────────────────────────────────────────────────────────────────────

    def fase_agregacao(self, frames_validos: int) -> dict | None:
        """
        Calcula a média de cada feature sobre todos os frames válidos.

        POR QUE USAR MÉDIA?
        ────────────────────────────────────────────────────────────────────────
        A média é o estimador de máxima verossimilhança para dados com ruído
        gaussiano — que é exatamente o que temos: features reais + ruído de
        sensor/iluminação/pixelização.

        Ao tirar a média de N frames, o erro padrão cai por √N:
          N=1  : erro = σ
          N=16 : erro = σ/4
          N=60 : erro = σ/7.7   ← muito mais estável que 1 frame

        CASO ESPECIAL: holes e is_hollow
        ────────────────────────────────────────────────────────────────────────
        holes e is_hollow são valores BINÁRIOS (0 ou 1 por frame).
        Sua média é uma proporção: 0.93 significa "93% dos frames detectaram furo".

        Usamos round() para converter de volta para 0 ou 1 antes de passar
        ao modelo, porque o modelo foi treinado com valores inteiros.

        Isso equivale a uma VOTAÇÃO POR MAIORIA:
          mean ≥ 0.5 → round() → 1 → "a maioria dos frames detectou furo"
          mean < 0.5 → round() → 0 → "a maioria dos frames NÃO detectou furo"

        Retorna:
            dict com as features agregadas, prontas para o modelo.
            None se não houver frames suficientes.
        """
        titulo("FASE 3 — AGREGAÇÃO ESTATÍSTICA")

        # ── Validação: frames suficientes? ────────────────────────────────────
        if frames_validos == 0:
            print("  [ERRO] Nenhum frame válido capturado.")
            print("  Verifique:")
            print("    1. O perfil está dentro do campo de visão da câmera?")
            print(f"   2. min_area={self.detector.min_area} px² — tente reduzir se o perfil for pequeno")
            print("    3. A iluminação está razoável?")
            return None

        frac_validos = frames_validos / self.n_frames
        if frac_validos < self.min_validos:
            print(
                f"  [AVISO] Apenas {frac_validos:.0%} dos frames foram válidos "
                f"(mínimo esperado: {self.min_validos:.0%})."
            )
            print("  O resultado pode ser impreciso. Recomenda-se capturar novamente")
            print("  em ambiente mais controlado (fundo escuro, iluminação difusa).")

        # ── Extrai os valores de cada feature como array numpy ────────────────
        #
        # .get(feature, 0) → segurança: se algum frame não tiver a feature,
        # usa 0 em vez de lançar KeyError.
        #
        # Resultado: cada variável abaixo é um array 1D com `frames_validos` valores.
        circularity_vals  = np.array([f.get("circularity",  0) for f in self._features_por_frame])
        aspect_ratio_vals = np.array([f.get("aspect_ratio", 0) for f in self._features_por_frame])
        holes_vals        = np.array([f.get("holes",        0) for f in self._features_por_frame])
        area_vals         = np.array([f.get("area",         0) for f in self._features_por_frame])
        is_hollow_vals    = np.array([f.get("is_hollow",    0) for f in self._features_por_frame])
        # material não varia entre frames — é constante (input do operador)
        material_val      = self.material

        # ── Calcula médias e desvios ───────────────────────────────────────────
        medias = {
            "circularity":  float(np.mean(circularity_vals)),
            "aspect_ratio": float(np.mean(aspect_ratio_vals)),
            "holes_media":  float(np.mean(holes_vals)),        # proporção bruta
            "holes":        int(round(np.mean(holes_vals))),   # voto de maioria
            "area":         float(np.mean(area_vals)),
            "is_hollow":    int(round(np.mean(is_hollow_vals))),
            "material":     material_val,
        }

        desvios = {
            "circularity":  float(np.std(circularity_vals)),
            "aspect_ratio": float(np.std(aspect_ratio_vals)),
            "holes":        float(np.std(holes_vals)),
            "area":         float(np.std(area_vals)),
            "is_hollow":    float(np.std(is_hollow_vals)),
        }

        # ── Print da tabela de features ───────────────────────────────────────
        print(f"  Frames usados na agregação: {frames_validos}\n")

        colunas = f"  {'FEATURE':<16} {'MÉDIA':>10}  {'DESVIO':>8}  {'MÍN':>8}  {'MÁX':>8}  OBSERVAÇÃO"
        print(colunas)
        print("  " + "-" * 75)

        features_para_tabela = [
            ("circularity",  circularity_vals,  medias["circularity"],  desvios["circularity"],
             f"→ {'REDONDO' if medias['circularity'] > 0.85 else 'QUADRADO/RET'} (limiar ~0.85)"),
            ("aspect_ratio", aspect_ratio_vals, medias["aspect_ratio"], desvios["aspect_ratio"],
             f"→ {'APROX QUADRADO' if 0.85 < medias['aspect_ratio'] < 1.15 else 'RETANGULAR/CHATO'}"),
            ("holes",        holes_vals,         medias["holes_media"],  desvios["holes"],
             f"→ {'OCO detectado' if medias['holes'] == 1 else 'NÃO detectado'} "
             f"(voto: {medias['holes_media']:.2f} → {medias['holes']})"),
            ("area",         area_vals,           medias["area"],         desvios["area"],
             "px²"),
            ("is_hollow",    is_hollow_vals,      float(medias["is_hollow"]), desvios["is_hollow"],
             f"→ {'OCO' if medias['is_hollow'] == 1 else 'MACIÇO'} (voto de maioria)"),
        ]

        for nome, vals, media, desvio, obs in features_para_tabela:
            print(
                f"  {nome:<16} {media:>10.4f}  {desvio:>8.4f}  "
                f"{vals.min():>8.3f}  {vals.max():>8.3f}  {obs}"
            )

        print(f"  {'material':<16} {material_val:>10}  {'—':>8}  {'—':>8}  "
              f"{'—':>8}  {MATERIAL_LABEL.get(material_val, '?')} (input do operador)")

        # ── Monta o dict final que vai para o modelo ───────────────────────────
        #
        # Este é o passo central do script: transformar N frames de features
        # brutas em UM único vetor de features médias, pronto para o modelo.
        #
        # ATENÇÃO: a chave "holes_media" é excluída — era só para o relatório.
        # O modelo espera exatamente as chaves do FEATURE_ORDER.

        dict_features = {
            "circularity":  round(medias["circularity"],  4),
            "aspect_ratio": round(medias["aspect_ratio"], 4),
            "holes":        medias["holes"],       # int (0 ou 1, voto de maioria)
            "area":         round(medias["area"],  2),
            "is_hollow":    medias["is_hollow"],   # int (0 ou 1, voto de maioria)
            "material":     material_val,          # int (código do material)
        }

        # ── Confirmação visual no terminal ────────────────────────────────────
        print()
        print("  ✓ Média calculada. Dict de features alocado em memória:")
        print()
        print("  features_agregadas = {")
        for chave, valor in dict_features.items():
            tipo = type(valor).__name__
            print(f"    '{chave}': {valor}  # {tipo}")
        print("  }")

        return dict_features

    # ──────────────────────────────────────────────────────────────────────────
    #  FASE 4 — PREDIÇÃO
    # ──────────────────────────────────────────────────────────────────────────

    def fase_predicao(self, features_agregadas: dict):
        """
        Passa o dict de features médias para o RandomForest e exibe o resultado.

        O modelo recebe UMA linha de features (o vetor médio) e retorna:
          • label         : nome da classe predita
          • confidence    : proporção de votos das 200 árvores para essa classe
          • low_confidence: flag se confiança < 70%
          • all_probas    : probabilidade de CADA classe possível
          • message       : mensagem formatada para o operador

        Por que a predição aqui é mais confiável que em 1 frame só?
        ────────────────────────────────────────────────────────────────────────
        No test.py, cada frame individual tem ruído alto.
        Aqui as features são a média de 60 frames — o vetor de entrada
        é muito mais estável e representativo do perfil real.
        A confiança da predição tende a ser significativamente maior.
        """
        titulo("FASE 4 — PREDIÇÃO DO MODELO")

        if self.classificador is None:
            print("  [PULADO] Modelo não carregado.")
            print("  Execute: python ml/training.py")
            return None

        # ── Chamada ao modelo ─────────────────────────────────────────────────
        # Classifier.predict() faz internamente:
        #   1. Ordena as features de acordo com FEATURE_ORDER
        #   2. Cria um DataFrame de 1 linha
        #   3. Chama clf.predict_proba() → votos das 200 árvores
        #   4. Retorna o resultado formatado
        resultado = self.classificador.predict(features_agregadas)

        if resultado.get("error"):
            print(f"  [ERRO] {resultado['error']}")
            return None

        label      = resultado["label"]
        conf_pct   = resultado["confidence_pct"]
        baixa_conf = resultado["low_confidence"]
        all_probas = resultado["all_probas"]

        # ── Output principal ──────────────────────────────────────────────────
        simbolo = "✓" if not baixa_conf else "⚠"
        print(f"  {simbolo}  {resultado['message']}")
        print()

        # Barra de confiança visual
        filled = int(conf_pct / 5)
        barra  = "█" * filled + "░" * (20 - filled)
        print(f"  Confiança : [{barra}] {conf_pct}%")
        print()

        # Probabilidades de todas as classes
        print("  Probabilidades por classe:")
        for cls, prob in sorted(all_probas.items(), key=lambda x: -x[1]):
            p_pct  = round(prob * 100)
            b      = "█" * int(p_pct / 4)
            marca  = " ← PREDITO" if cls == label else ""
            print(f"    {cls:<35s}: {b:<25s} {p_pct:3d}%{marca}")

        print()

        # ── Aviso de confiança baixa ───────────────────────────────────────────
        if baixa_conf:
            print("  ─── ATENÇÃO ─────────────────────────────────────────────")
            print(f"  Confiança abaixo de 70%. Possíveis causas:")
            print("    • Perfil posicionado de lado (câmera deve ver a ponta)")
            print("    • Iluminação muito variável durante a captura")
            print("    • Perfil de geometria atípica (ex: tubo oval, amassado)")
            print("    • Poucos frames válidos — tente capturar novamente")

        return resultado

    # ──────────────────────────────────────────────────────────────────────────
    #  MÉTODO PRINCIPAL: run()
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> dict | None:
        """
        Executa o pipeline completo das 4 fases em sequência.

        Retorna o resultado da predição ou None se abortado/erro.
        """
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║   CLASSIFICADOR DE PERFIS METÁLICOS — MECALD / SENAI   ║")
        print("╚══════════════════════════════════════════════════════════╝")

        resultado_final = None

        try:
            # Conecta a câmera
            self.camera.connect()
            self.camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Janelas redimensionáveis 
            for nome in [ 
                        "Captura de perfil - Posicionamento",
                        "Captura de perfil - Coletando",
                        "Captura de perfil - Resultado", 
                        "Limiarização (CLAHE + Adaptativo)"
                        ]: 
                cv2.namedWindow(nome,cv2.WINDOW_NORMAL)
                
                 
            cv2.resizeWindow("Captura de Perfil - Posicionamento", 1280, 720)
            cv2.resizeWindow("Captura de Perfil - Coletando", 1280, 720)           
            cv2.resizeWindow("Captura de Perfil - Resultado", 1280, 720)           
            cv2.resizeWindow("Limiarização (CLAHE + Adaptativo)", 1280, 720)           

            print(f"\n  Câmera conectada.  Frames: {self.n_frames}  Countdown: {self.countdown}s\n")

            # ── FASE 1: COUNTDOWN ──────────────────────────────────────────────
            ok = self.fase_countdown()
            if not ok:
                return None

            # ── FASE 2: CAPTURA ────────────────────────────────────────────────
            frames_validos = self.fase_captura()

            # ── FASE 3: AGREGAÇÃO ──────────────────────────────────────────────
            features_agregadas = self.fase_agregacao(frames_validos)
            if features_agregadas is None:
                return None

            # ── FASE 4: PREDIÇÃO ───────────────────────────────────────────────
            resultado_final = self.fase_predicao(features_agregadas)

            # Mostra resultado final na janela da câmera por 3 segundos
            if resultado_final:
                frame = self.camera.read_frame()
                desenhar_resultado_frame(frame, resultado_final)
                cv2.imshow("Captura de Perfil — Resultado", frame)
                while True: 
                    cv2.imshow("Captura de Perfil - Resultado", frame)
                    
                    if cv2.waitKey(3000) & 0xFF == 27: 
                        break 
        finally:
            # Garante que a câmera é liberada mesmo se houver exceção
            self.camera.release()

        titulo("FIM")
        print("  Câmera liberada. Janelas encerradas.")
        print()

        return resultado_final


# ════════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════════

def perguntar_material() -> int:
    """
    Prompt interativo para o operador selecionar o material do perfil.
    Retorna o código numérico do material selecionado.
    """
    print()
    print("  Selecione o material do perfil:")
    print("    0 → Aço Carbono")
    print("    1 → Inox")
    print("    2 → Alumínio")
    print()

    while True:
        try:
            entrada = input("  Material [0/1/2]: ").strip()
            codigo  = int(entrada)
            if codigo in MATERIAL_MAP.values():
                return codigo
            print("  Digite 0, 1 ou 2.")
        except (ValueError, KeyboardInterrupt):
            print("  Usando padrão: 0 (Aço Carbono)")
            return 0


# ════════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — PARSING DE ARGUMENTOS
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Captura e classifica um perfil metálico usando câmera + ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python capturar_perfil.py
  python capturar_perfil.py --material inox --countdown 8 --frames 80
  python capturar_perfil.py --material aco_carbono --min-area 1000
        """
    )

    parser.add_argument(
        "--material",
        choices=["aco_carbono", "inox", "aluminio"],
        default=None,
        help="Material do perfil. Se omitido, será perguntado interativamente."
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        metavar="SEGUNDOS",
        help="Segundos de countdown antes da captura. Padrão: 5"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        metavar="N",
        help="Número de frames a capturar. Padrão: 60"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="ID",
        help="Índice da câmera. Padrão: 0"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=2000,
        metavar="PX2",
        help="Área mínima do contorno em px². Padrão: 2000. Reduza se perfil pequeno."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="ml/",
        help="Pasta onde estão os arquivos do modelo. Padrão: ml/"
    )

    args = parser.parse_args()

    # ── Material: argumento CLI ou prompt interativo ───────────────────────────
    if args.material is not None:
        material_codigo = MATERIAL_MAP[args.material]
        print(f"\n  Material: {args.material} (código {material_codigo})")
    else:
        material_codigo = perguntar_material()

    # ── Instancia e executa ────────────────────────────────────────────────────
    captura = CapturaEClassificador(
        n_frames   = args.frames,
        countdown  = args.countdown,
        camera_id  = args.camera,
        material   = material_codigo,
        min_area   = args.min_area,
        model_dir  = args.model_dir,
    )

    resultado = captura.run()

    # ── Exit code: 0 se predição com alta confiança, 1 caso contrário ─────────
    if resultado and not resultado.get("low_confidence", True):
        sys.exit(0)
    else:
        sys.exit(1)
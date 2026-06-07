"""
gerar_dataset_v2.py
════════════════════════════════════════════════════════════════════════════════
Dataset sintético — 4 classes de perfis metálicos — Mecald / SENAI

EXECUTE ESTE ARQUIVO para gerar o CSV:
    python ml/gerar_dataset_v2.py

ARQUIVO GERADO:
    ml/perfis_dataset_v2.csv   (600 amostras, 150 por classe)

════════════════════════════════════════════════════════════════════════════════
  AS 4 CLASSES E SUA BASE FÍSICA
════════════════════════════════════════════════════════════════════════════════

  CLASSE 1 — tubo_redondo_oco
  ─────────────────────────────────────────────────────────────────
  Perfil com seção transversal CIRCULAR e vazio interno (furo).
  Produtos reais: tubos estruturais, hidráulicos, tubos de irrigação.

  CLASSE 2 — tubo_quadrado_oco
  ─────────────────────────────────────────────────────────────────
  Perfil com seção QUADRADA ou RETANGULAR e vazio interno.
  Produtos reais: tubos estruturais quadrados (30×30, 50×50 mm),
  tubos retangulares (40×80, 50×100 mm).

  CLASSE 3 — tubo_redondo_macico
  ─────────────────────────────────────────────────────────────────
  Barra redonda SEM furo interno — completamente sólida.
  Produtos reais: barras de transmissão, eixos, barras CA 50/60.

  CLASSE 4 — tubo_quadrado_macico
  ─────────────────────────────────────────────────────────────────
  Barra quadrada, retangular ou chata — completamente sólida.
  Produtos reais: barra chata (20×5 mm), barra quadrada (20×20 mm),
  cantoneiras em barra.

════════════════════════════════════════════════════════════════════════════════
  MAPA DE FEATURES → LABELS
════════════════════════════════════════════════════════════════════════════════

  Como as features discriminam as 4 classes:

  ┌─────────────────────┬─────────────┬───────────────┬───────────┬──────────┐
  │ FEATURE             │ RDO_OC      │ QUAD_OC        │ RDO_MAC   │ QUAD_MAC │
  ├─────────────────────┼─────────────┼───────────────┼───────────┼──────────┤
  │ circularity         │ 0.88 – 1.00 │ 0.45 – 0.82   │ 0.87–1.00 │ 0.30–0.82│
  │ aspect_ratio        │ 0.90 – 1.10 │ 0.40 – 1.05   │ 0.88–1.12 │ 0.15–1.05│
  │ holes               │ 0 ou 1      │ 0 ou 1        │ SEMPRE 0  │ SEMPRE 0 │
  │ area (px²)          │ 100 – 3500  │ 150 – 4500    │ 250–12000 │ 300–18000│
  │ is_hollow           │ 0 ou 1      │ 0 ou 1        │ SEMPRE 0  │ SEMPRE 0 │
  │ material            │ 0 / 1 / 2   │ 0 / 1 / 2     │ 0/1/2     │ 0/1/2    │
  └─────────────────────┴─────────────┴───────────────┴───────────┴──────────┘

  ÁRVORE DE DECISÃO LÓGICA (o que o RF aprende internamente):
  ─────────────────────────────────────────────────────────────────
  circularity > 0.84?
  │
  ├─ SIM (redondo)
  │   │
  │   is_hollow == 1  ou  holes > 0?
  │   ├─ SIM → tubo_redondo_oco
  │   └─ NÃO → area < limiar_area?
  │             ├─ SIM → tubo_redondo_oco (furo não detectado)
  │             └─ NÃO → tubo_redondo_macico
  │
  └─ NÃO (quadrado/retangular)
      │
      is_hollow == 1  ou  holes > 0?
      ├─ SIM → tubo_quadrado_oco
      └─ NÃO → area < limiar_area?
                ├─ SIM → tubo_quadrado_oco (furo não detectado)
                └─ NÃO → tubo_quadrado_macico

  Por que 5% de erro acontece?
  ─────────────────────────────────────────────────────────────────
  Os erros estão concentrados em casos onde:
    • O furo do tubo OC O NÃO FOI detectado (holes=0 por iluminação/ângulo)
    • A área do tubo oco grande sobrepõe a área de barra maciça pequena
  → O modelo fica incerto entre oco e maciço de mesma forma geométrica.
  → Solução prática: garantir posicionamento correto (câmera na ponta do tubo).

════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N   = 150   # amostras por classe → 600 total

def noise(arr, sigma):
    return arr + rng.normal(0, sigma, size=len(arr))

def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ════════════════════════════════════════════════════════════════════════════
#  CLASSE 1 — tubo_redondo_oco
# ════════════════════════════════════════════════════════════════════════════
#
#  POR QUE circularity ≈ 0.88–1.00?
#  ───────────────────────────────────────────────────────────────────────────
#  Fórmula: circularity = 4π × A / P²
#  Para círculo perfeito: A = πr², P = 2πr
#    → circularity = 4π × πr² / (2πr)² = 4π² r² / 4π² r² = 1.000
#  Na prática, pixelização e leve deformação reduzem para 0.88–0.99.
#
#  POR QUE aspect_ratio ≈ 1.0?
#  ───────────────────────────────────────────────────────────────────────────
#  Bounding box de um círculo: largura ≈ altura → w/h ≈ 1.0
#  Variação de 4% simula: ângulo não perfeitamente ortogonal, leve ovaling.
#
#  POR QUE holes ≈ 92% positivo (não 100%)?
#  ───────────────────────────────────────────────────────────────────────────
#  O cv2.findContours detecta o furo quando há contraste suficiente entre
#  a parede do tubo e o interior. Falha quando:
#    - Parede muito grossa (D pequeno, t grande) → furo minúsculo em px
#    - Iluminação entra pelo furo e iguala o brilho da parede
#    - Ângulo levemente oblíquo oculta o furo
#  → 8% de falso negativo é conservador e realista.
#
#  POR QUE area ≈ 100–3500 px²?
#  ───────────────────────────────────────────────────────────────────────────
#  O OpenCV mede a área do contorno EXTERNO, que para tubo oco é a área
#  do ANEL (seção transversal da parede, não da seção total).
#  Área do anel = π(R² - r²) = π(2Rt - t²) ≈ 2πRt
#  Exemplo: tubo 48mm OD, 2mm parede → A ≈ 289 mm²
#  Com escala 1.5 px/mm: A_px ≈ 289 × 1.5 ≈ 434 px²
#  Tubos grandes com parede grossa chegam a ~3500 px².

diameters_roc = rng.choice(
    [21.3, 26.7, 33.4, 42.2, 48.3, 60.3, 73.0, 88.9, 101.6, 114.3], size=N
)
wall_t_roc   = rng.uniform(1.5, 4.0, N)
R_roc        = diameters_roc / 2
r_roc        = R_roc - wall_t_roc
area_ring    = np.pi * (R_roc**2 - r_roc**2)

circ_roc     = clamp(noise(np.full(N, 0.95), 0.03), 0.88, 1.00)
ar_roc       = clamp(noise(np.ones(N), 0.04), 0.90, 1.10)
holes_roc    = rng.choice([0, 1], size=N, p=[0.08, 0.92])
area_roc     = clamp(noise(area_ring * 1.5, area_ring * 0.08), 100, 3500)
mat_roc      = rng.choice([0, 1, 2], size=N, p=[0.60, 0.25, 0.15])

rows_roc = [{
    "circularity":  round(float(circ_roc[i]),  4),
    "aspect_ratio": round(float(ar_roc[i]),    4),
    "holes":        int(holes_roc[i]),
    "area":         round(float(area_roc[i]),  2),
    "is_hollow":    int(holes_roc[i]),
    "material":     int(mat_roc[i]),
    "label":        "tubo_redondo_oco",
} for i in range(N)]


# ════════════════════════════════════════════════════════════════════════════
#  CLASSE 2 — tubo_quadrado_oco
# ════════════════════════════════════════════════════════════════════════════
#
#  POR QUE circularity ≈ 0.45–0.82?
#  ───────────────────────────────────────────────────────────────────────────
#  Para quadrado perfeito: A = a², P = 4a → circ = 4π·a²/16a² = π/4 ≈ 0.785
#  Para retângulo 1:2: circ ≈ 2π/9 ≈ 0.698
#  Para retângulo 1:3: circ ≈ 3π/16 ≈ 0.589
#  → Quanto mais retangular, menor a circularity.
#  → Nunca ultrapassa 0.82 (não confunde com redondo).
#
#  POR QUE aspect_ratio ≈ 0.40–1.05?
#  ───────────────────────────────────────────────────────────────────────────
#  Tubos quadrados: w ≈ h → ar ≈ 1.0 (idêntico ao redondo!)
#  Tubos retangulares: w < h → ar < 1.0 (ex: 40×80mm → ar = 0.5)
#  → aspect_ratio SOZINHO não separa redondo de quadrado (ambos ≈ 1.0).
#    Precisa da circularity para isso.
#
#  POR QUE holes ≈ 85% positivo (menor que redondo)?
#  ───────────────────────────────────────────────────────────────────────────
#  Tubo quadrado tem seção interna quadrada, também com cantos.
#  O threshold adaptativo tem mais dificuldade em detectar cantos internos
#  do que um furo circular. Taxa de falso negativo maior: 15%.
#
#  POR QUE area ≈ 150–4500 px²?
#  ───────────────────────────────────────────────────────────────────────────
#  Área da parede = W×H - (W-2t)×(H-2t)
#  Exemplo 50×50mm, 2mm parede: A = 2500 - 2116 = 384 mm² → ~576 px²
#  Tubos retangulares grandes chegam a ~4500 px².

n_sq  = 90
n_rec = 60
lado_sq  = rng.choice([20, 25, 30, 40, 50, 60, 75, 80, 100], size=n_sq).astype(float)
w_rec    = rng.choice([20, 25, 30, 40, 50, 60, 80], size=n_rec).astype(float)
h_rec    = w_rec * rng.uniform(1.3, 2.0, n_rec)

w_qoc = np.concatenate([lado_sq, w_rec])
h_qoc = np.concatenate([lado_sq, h_rec])
wall_t_qoc   = rng.uniform(1.5, 3.0, N)
ratio_wh_qoc = np.minimum(w_qoc, h_qoc) / np.maximum(w_qoc, h_qoc)

circ_qoc  = clamp(noise(np.pi/4 * (0.55 + 0.45*ratio_wh_qoc), 0.03), 0.45, 0.82)
ar_qoc    = clamp(noise(w_qoc / h_qoc, 0.04), 0.40, 1.05)
holes_qoc = rng.choice([0, 1], size=N, p=[0.15, 0.85])
area_wall = w_qoc*h_qoc - (w_qoc-2*wall_t_qoc)*(h_qoc-2*wall_t_qoc)
area_qoc  = clamp(noise(area_wall * 1.5, area_wall * 0.08), 150, 4500)
mat_qoc   = rng.choice([0, 1, 2], size=N, p=[0.70, 0.15, 0.15])

rows_qoc = [{
    "circularity":  round(float(circ_qoc[i]),  4),
    "aspect_ratio": round(float(ar_qoc[i]),    4),
    "holes":        int(holes_qoc[i]),
    "area":         round(float(area_qoc[i]),  2),
    "is_hollow":    int(holes_qoc[i]),
    "material":     int(mat_qoc[i]),
    "label":        "tubo_quadrado_oco",
} for i in range(N)]


# ════════════════════════════════════════════════════════════════════════════
#  CLASSE 3 — tubo_redondo_macico
# ════════════════════════════════════════════════════════════════════════════
#
#  POR QUE circularity ≈ 0.87–1.00? (IDÊNTICA ao tubo_redondo_oco!)
#  ───────────────────────────────────────────────────────────────────────────
#  A seção transversal de uma barra redonda É um círculo — igual ao tubo.
#  O contorno externo tem a mesma forma. Circularity não consegue distinguir.
#  → Por isso holes e area são ESSENCIAIS para esta separação.
#
#  POR QUE holes = 0 SEMPRE? (certeza física, não probabilidade)
#  ───────────────────────────────────────────────────────────────────────────
#  Barra maciça não tem espaço vazio interno.
#  O cv2.findContours com RETR_TREE só retorna filhos se há CONTORNO INTERNO.
#  Sem furo → sem contorno interno → holes = 0, 100% das vezes.
#  → Esta é a feature mais confiável para distinguir maciço de oco.
#
#  POR QUE area ≈ 250–12000 px²? (MUITO MAIOR que o oco)
#  ───────────────────────────────────────────────────────────────────────────
#  O contorno externo de uma barra sólida engloba TODA a seção → πr²
#  Vs o tubo oco de mesmo diâmetro: apenas a área da parede (anel).
#
#  Comparação direta (mesmos diâmetros):
#    Tubo oco  48mm, 2mm parede → A_oco   = π(24² - 22²) =  289 mm²  → ~433 px²
#    Barra mac 48mm              → A_maci  = π(24²)        = 1810 mm² → 2715 px²
#    Ratio: maciço / oco ≈ 6.3×
#
#  Com diâmetros variando (barra de 10mm vs tubo de 114mm), os ranges se
#  sobrepõem um pouco, mas a distribuição média é claramente maior.
#  O modelo usa a COMBINAÇÃO holes=0 + area grande para máxima certeza.

diam_rma      = rng.choice([10, 12, 16, 20, 25, 30, 40, 50, 60, 75, 90], size=N).astype(float)
area_circ     = np.pi * (diam_rma / 2)**2

circ_rma      = clamp(noise(np.full(N, 0.95), 0.03), 0.87, 1.00)
ar_rma        = clamp(noise(np.ones(N), 0.04), 0.88, 1.12)
holes_rma     = np.zeros(N, dtype=int)          # SEMPRE 0 — certeza física
is_hollow_rma = np.zeros(N, dtype=int)          # SEMPRE 0
area_rma      = clamp(noise(area_circ * 1.5, area_circ * 0.08), 250, 12000)
mat_rma       = rng.choice([0, 1, 2], size=N, p=[0.75, 0.15, 0.10])

rows_rma = [{
    "circularity":  round(float(circ_rma[i]),  4),
    "aspect_ratio": round(float(ar_rma[i]),    4),
    "holes":        int(holes_rma[i]),
    "area":         round(float(area_rma[i]),  2),
    "is_hollow":    int(is_hollow_rma[i]),
    "material":     int(mat_rma[i]),
    "label":        "tubo_redondo_macico",
} for i in range(N)]


# ════════════════════════════════════════════════════════════════════════════
#  CLASSE 4 — tubo_quadrado_macico
# ════════════════════════════════════════════════════════════════════════════
#
#  POR QUE circularity ≈ 0.30–0.82? (mais ampla que o oco quadrado)
#  ───────────────────────────────────────────────────────────────────────────
#  Inclui "barra chata" (chato = flat bar): proporção 1:3, 1:5, 1:6
#  Barra chata 100×10mm → circ ≈ 4π·(100×10)/(2×110)² ≈ 0.32
#  Isso cria uma "cauda" em circularity baixa inexistente no oco quadrado.
#
#  POR QUE aspect_ratio chega a 0.15?
#  ───────────────────────────────────────────────────────────────────────────
#  Barra chata: produto comum em metalúrgica.
#  Dimensões típicas: 20×5, 30×6, 50×8, 100×12 mm.
#  Para 50×8mm: ar = 8/50 = 0.16.
#  O tubo quadrado oco NUNCA tem parede tão fina que dê essa proporção.
#  → aspect_ratio baixo é um indicador forte de "chato maciço".
#
#  POR QUE holes = 0 SEMPRE?
#  ───────────────────────────────────────────────────────────────────────────
#  Mesmo raciocínio do tubo redondo maciço: sem espaço interno → sem contorno
#  filho na hierarquia do findContours.
#
#  POR QUE area ≈ 300–18000 px²? (maior de todas as classes)
#  ───────────────────────────────────────────────────────────────────────────
#  Barra 100×100mm: A = 10000 mm² → ~15000 px² (muito maior que qualquer oco)
#  Mesmo a barra chata 100×12mm: A = 1200 mm² → ~1800 px²
#  O range é o maior de todas as 4 classes.

n_sq2  = 75
n_rec2 = 45
n_flat = 30

lado_sq2 = rng.choice([10, 12, 15, 20, 25, 30, 40, 50, 60, 75], size=n_sq2).astype(float)
w_rec2   = rng.choice([20, 25, 30, 40, 50, 60, 80], size=n_rec2).astype(float)
h_rec2   = w_rec2 * rng.uniform(1.2, 2.5, n_rec2)
w_flat   = rng.choice([20, 25, 30, 40, 50, 60, 75, 100], size=n_flat).astype(float)
h_flat   = w_flat * rng.uniform(2.5, 6.0, n_flat)   # barra chata: proporção 1:2.5 a 1:6

w_qma        = np.concatenate([lado_sq2, w_rec2, w_flat])
h_qma        = np.concatenate([lado_sq2, h_rec2, h_flat])
ratio_wh_qma = np.minimum(w_qma, h_qma) / np.maximum(w_qma, h_qma)

circ_qma      = clamp(noise(np.pi/4 * (0.50 + 0.50*ratio_wh_qma), 0.03), 0.30, 0.82)
ar_qma        = clamp(noise(w_qma / h_qma, 0.04), 0.15, 1.05)
holes_qma     = np.zeros(N, dtype=int)          # SEMPRE 0
is_hollow_qma = np.zeros(N, dtype=int)          # SEMPRE 0
area_solid    = w_qma * h_qma
area_qma      = clamp(noise(area_solid * 1.5, area_solid * 0.08), 300, 18000)
mat_qma       = rng.choice([0, 1, 2], size=N, p=[0.80, 0.10, 0.10])

rows_qma = [{
    "circularity":  round(float(circ_qma[i]),  4),
    "aspect_ratio": round(float(ar_qma[i]),    4),
    "holes":        int(holes_qma[i]),
    "area":         round(float(area_qma[i]),  2),
    "is_hollow":    int(is_hollow_qma[i]),
    "material":     int(mat_qma[i]),
    "label":        "tubo_quadrado_macico",
} for i in range(N)]


# ─── Consolidar, embaralhar e salvar ─────────────────────────────────────────
df = pd.DataFrame(rows_roc + rows_qoc + rows_rma + rows_qma)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = Path(__file__).parent / "perfis_dataset_v2.csv"
df.to_csv(out_path, index=False)

# ─── Relatório de validação ───────────────────────────────────────────────────
print("=" * 68)
print("  DATASET GERADO — PERFIS_DATASET_V2.CSV")
print("=" * 68)
print(f"  Total de amostras : {len(df)}")
print(f"  Classes           : {df['label'].nunique()}")
print()
print("  Distribuição de classes:")
for cls, cnt in df["label"].value_counts().sort_index().items():
    print(f"    {cls:35s}: {cnt}")
print()
print("  Médias por classe e feature (mostra separabilidade):")
print(
    df.groupby("label")[
        ["circularity","aspect_ratio","holes","area","is_hollow"]
    ].mean().round(3).to_string()
)
print()
print("  Ranges de area (principal diferenciador oco vs maciço):")
for cls in sorted(df["label"].unique()):
    sub = df[df["label"] == cls]["area"]
    print(f"    {cls:35s}: {sub.min():.0f} – {sub.max():.0f}  (μ={sub.mean():.0f})")
print()
print(f"  ✓ Arquivo salvo em: {out_path}")
print()
print("  PRÓXIMO PASSO:")
print("    Substitua 'perfis_dataset.csv' por 'perfis_dataset_v2.csv'")
print("    no treinamento e rode: python ml/training_v2.py")


if __name__ == "__main__":
    pass  # tudo já executou no nível de módulo acima
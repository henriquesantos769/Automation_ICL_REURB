try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    import pymupdf as fitz

import pandas as pd
from pathlib import Path
import math

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
pdf_path = Path(r"C:\automation_vec\mapa geográficos\São José - Curvelo.pdf")
output_xlsx = Path("São José - Curvelo.xlsx")
MAX_DIST_IMOVEL = 15   # raio em px para associar IMÓVEL ao LOTE
MAX_DIST_RES = 15      # raio em px para associar RES ao LOTE


# Cores de referência (RGB) para cada tipo
COLOR_REF = {
    "QUADRA": (255,   0,   0),   # vermelho
    "LOTE":   (  0,   0, 255),   # azul
    "IMOVEL": (  0, 180,   0),   
    "RES":    (255, 165,   0),   
}

# Limiares (quanto maior, mais permissivo). Ajuste IMOVEL se algum verde não pegar.
THRESH = {
    "QUADRA": 140.0,
    "LOTE":   160.0,
    "IMOVEL": 175.0,   # << se um verde ainda escapar, aumente um pouco (ex.: 185)
    "RES":    170.0,
}

# -----------------------------
# FUNÇÕES
# -----------------------------
def int_to_rgb(val: int):
    r = (val >> 16) & 255
    g = (val >> 8) & 255
    b = val & 255
    return r, g, b

def color_distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def classify_by_nearest_color(rgb):
    """Classifica pelo protótipo de cor mais próximo (com limiar)."""
    best_typ, best_d = None, 1e9
    for typ, ref in COLOR_REF.items():
        d = color_distance(rgb, ref)
        if d < best_d:
            best_typ, best_d = typ, d
    if best_typ and best_d <= THRESH[best_typ]:
        return best_typ
    return "OTHER"

# -----------------------------
# EXTRAÇÃO DO PDF
# -----------------------------
doc = fitz.open(str(pdf_path))
records = []

for page_num, page in enumerate(doc, start=1):
    blocks = page.get_text("dict")["blocks"]
    for blk in blocks:  # 'blk' para não conflitar com canal 'b' do RGB
        if "lines" not in blk:
            continue
        for l in blk["lines"]:
            for s in l["spans"]:
                text = s["text"].strip()

                # Se o span tiver "nº 210", extrai só os dígitos;
                # mantém o seu comportamento de ignorar textos sem dígitos:
                if not any(ch.isdigit() for ch in text):
                    continue
                digits_only = "".join(ch for ch in text if ch.isdigit())
                if not digits_only:
                    continue

                r, g, bb = int_to_rgb(s["color"])  # 'bb' para não colidir com 'blk'
                typ = classify_by_nearest_color((r, g, bb))
                if typ == "OTHER":
                    continue

                x0, y0, x1, y1 = s["bbox"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                records.append({
                    "page": page_num,
                    "text": digits_only,
                    "type": typ,
                    "x": cx,
                    "y": cy
                })

df_raw = pd.DataFrame(records)

# -----------------------------
# SEPARAR POR TIPOS
# -----------------------------
quadras = df_raw[df_raw["type"] == "QUADRA"]
lotes   = df_raw[df_raw["type"] == "LOTE"]
imoveis = df_raw[df_raw["type"] == "IMOVEL"]
resids  = df_raw[df_raw["type"] == "RES"]

# -----------------------------
# ASSOCIAÇÃO LOTE ↔ QUADRA / IMÓVEL / RESIDENCIAL
# -----------------------------
rows = []
for _, l in lotes.iterrows():
    # QUADRA mais próxima (sempre tenta preencher)
    qpage = quadras[quadras["page"] == l["page"]]
    quadra_val = None
    if not qpage.empty:
        dists = ((qpage["x"] - l["x"]) ** 2 + (qpage["y"] - l["y"]) ** 2) ** 0.5
        quadra_val = qpage.iloc[dists.argmin()]["text"]

    # IMÓVEL (verde) se dentro da distância
    ipage = imoveis[imoveis["page"] == l["page"]]
    imovel_val = None
    if not ipage.empty:
        dists = ((ipage["x"] - l["x"]) ** 2 + (ipage["y"] - l["y"]) ** 2) ** 0.5
        min_idx = dists.argmin()
        if dists.iloc[min_idx] <= MAX_DIST_IMOVEL:
            imovel_val = ipage.iloc[min_idx]["text"]

    # RESIDENCIAL (laranja) se dentro da distância (opcional)
    rpage = resids[resids["page"] == l["page"]]
    res_val = None
    if not rpage.empty:
        dists = ((rpage["x"] - l["x"]) ** 2 + (rpage["y"] - l["y"]) ** 2) ** 0.5
        min_idx = dists.argmin()
        if dists.iloc[min_idx] <= MAX_DIST_RES:
            res_val = rpage.iloc[min_idx]["text"]

    # ✅ Só mantém linhas com IMÓVEL (plus)
    if imovel_val is not None:
        rows.append({
            "LOTE": l["text"],
            "QUADRA": quadra_val,
            "IMOVEL (plus)": imovel_val,
            "nº Residencial": res_val
        })

df = pd.DataFrame(rows).drop_duplicates()

# -----------------------------
# SALVAR RESULTADO
# -----------------------------
df.to_excel(output_xlsx, index=False)
print(f"✅ Planilha gerada: {output_xlsx}")


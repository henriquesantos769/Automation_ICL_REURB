# -*- coding: utf-8 -*-
try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    import pymupdf as fitz

import argparse
import math
from pathlib import Path
import pandas as pd

# =========================
# CONFIG PADRÃO
# =========================
DEFAULT_MAX_DIST_IMOVEL = 15
DEFAULT_MAX_DIST_RES    = 15

COLOR_REF = {
    "QUADRA": (255,   0,   0),  # vermelho
    "LOTE":   (  0,   0, 255),  # azul
    "IMOVEL": (  0, 255,   0),  # verde
    "RES":    (255, 127,   0),  # laranja
}
THRESH = {
    "QUADRA": 140.0,
    "LOTE":   160.0,
    "IMOVEL": 175.0,
    "RES":    170.0,
}

# ======== CONFIG EXTRA (ajuste fino) ========
# mínimo de LOTEs que um polígono deve conter para ser aceito como "região da quadra"
DEFAULT_MIN_LOTES_IN_POLY = 8
# quantil usado p/ “pegar os LOTEs mais próximos do rótulo” ao criar a caixa
DEFAULT_FALLBACK_LOTE_QUANTILE = 0.35
# buffers em px para a caixa (expandir um pouco o retângulo)
DEFAULT_FALLBACK_PAD_INNER = 20
DEFAULT_FALLBACK_PAD_EXTRA = 100   # reforço de raio p/ selecionar LOTEs

# =========================
# UTILS
# =========================
def int_to_rgb(val: int):
    r = (val >> 16) & 255
    g = (val >> 8) & 255
    b = val & 255
    return r, g, b

def color_distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def classify_by_nearest_color(rgb):
    best_typ, best_d = None, 1e9
    for typ, ref in COLOR_REF.items():
        d = color_distance(rgb, ref)
        if d < best_d:
            best_typ, best_d = typ, d
    if best_typ and best_d <= THRESH[best_typ]:
        return best_typ
    return "OTHER"

def point_in_poly(x, y, poly):
    """Ray casting. poly: [(x,y), ...] (fechado ou não)."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    x0, y0 = poly[0]
    for i in range(1, n+1):
        x1, y1 = poly[i % n]
        cond = ((y0 > y) != (y1 > y)) and \
               (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
        if cond:
            inside = not inside
        x0, y0 = x1, y1
    return inside

def rect_contains_point(rect, x, y):
    x0, y0, x1, y1 = rect
    return (x0 <= x <= x1) and (y0 <= y <= y1)

def bbox_union(rects, pad=0):
    """rects: list of (x0,y0,x1,y1)."""
    if not rects:
        return None
    x0 = min(r[0] for r in rects) - pad
    y0 = min(r[1] for r in rects) - pad
    x1 = max(r[2] for r in rects) + pad
    y1 = max(r[3] for r in rects) + pad
    return (x0, y0, x1, y1)

def poly_area(poly):
    area = 0.0
    n = len(poly)
    if n < 3:
        return 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

# =========================
# VETORES → POLÍGONOS (tuple-aware)
# =========================
def extract_polygons_from_drawings(page, eps=1e-3):
    """
    Extrai polígonos fechados a partir dos vetores de desenho da página.
    Compatível com PyMuPDF que retorna items em tuplas, ex: ('l', Point(...), Point(...)).
    Também captura retângulos ('re') tanto no nível do drawing quanto no item.
    Retorna: [{"polygon": [(x,y),...], "stroke": (r,g,b) ou None, "width": float, "fill": (r,g,b) ou None}, ...]
    """
    polys = []
    drawings = page.get_drawings()

    for d in drawings:
        items = d.get("items", [])
        stroke = d.get("color")
        width  = d.get("width", 1.0)
        fill   = d.get("fill")
        closed_flag = d.get("closePath", False)

        # Caso 1: retângulo direto no drawing
        r = d.get("rect")
        if r is not None and d.get("type") == "re":
            poly = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
            polys.append({"polygon": poly, "stroke": stroke, "width": width, "fill": fill})
            continue

        # Caso 2: path composto por segmentos nos items
        poly_pts = []
        for it in items:
            if not isinstance(it, tuple) or not it:
                continue
            cmd = it[0]

            # Segmentos de linha: ('l', p0, p1)
            if cmd == 'l' and len(it) >= 3:
                p0, p1 = it[1], it[2]
                if not poly_pts:
                    poly_pts.append((p0.x, p0.y))
                poly_pts.append((p1.x, p1.y))

            # Retângulo como item: ('re', Rect)
            elif cmd == 're' and len(it) >= 2:
                rect = it[1]
                poly = [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)]
                polys.append({"polygon": poly, "stroke": stroke, "width": width, "fill": fill})

            # (Opcional) outros comandos ('m','c','h'...) podem ser tratados aqui se necessário

        # Fechamento do caminho: se closePath=True OU se primeiro≈último
        if len(poly_pts) >= 4:
            x0, y0 = poly_pts[0]
            xn, yn = poly_pts[-1]
            if closed_flag or math.hypot(xn - x0, yn - y0) < eps:
                polys.append({"polygon": poly_pts, "stroke": stroke, "width": width, "fill": fill})

    return polys

# =========================
# EXTRAÇÃO DE SPANS
# =========================
def extract_spans(page_num, page):
    recs = []
    blocks = page.get_text("dict")["blocks"]
    for blk in blocks:
        if "lines" not in blk:
            continue
        for l in blk["lines"]:
            for s in l["spans"]:
                text = s["text"].strip()
                # mantém somente spans com dígitos (Quadra "01", lote "12", etc.)
                if not any(ch.isdigit() for ch in text):
                    continue
                digits_only = "".join(ch for ch in text if ch.isdigit())
                if not digits_only:
                    continue
                r, g, b = int_to_rgb(s["color"])
                typ = classify_by_nearest_color((r, g, b))
                if typ == "OTHER":
                    continue
                x0, y0, x1, y1 = s["bbox"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                recs.append({
                    "page": page_num,
                    "text": digits_only,
                    "type": typ,
                    "x": cx, "y": cy,
                    "bbox": (x0, y0, x1, y1),
                })
    return recs

# =========================
# REGIÕES POR QUADRA (poly OU caixa)
# =========================
def choose_best_poly_for_quadra(label_x, label_y, polys, lotes_df):
    """
    Entre todos os polígonos que contêm o rótulo, escolhe o de MAIOR ÁREA.
    Retorna (poly, qtd_lotes_dentro).
    """
    candidates = []
    for p in polys:
        poly = p["polygon"]
        if point_in_poly(label_x, label_y, poly):
            candidates.append(poly)
    if not candidates:
        return None, 0

    best = max(candidates, key=poly_area)

    # conta quantos LOTEs caem dentro
    if lotes_df is None or lotes_df.empty:
        return best, 0
    cnt = 0
    for _, l in lotes_df.iterrows():
        if point_in_poly(l["x"], l["y"], best):
            cnt += 1
    return best, cnt

def make_fallback_box_for_quadra(qx, qy, lotes_df, lote_quantile, pad_inner, pad_extra):
    """
    Cria uma caixa a partir dos LOTEs mais próximos do rótulo da quadra,
    usando um quantil de distância + um reforço de raio.
    """
    if lotes_df is None or lotes_df.empty:
        return None
    d = ((lotes_df["x"] - qx) ** 2 + (lotes_df["y"] - qy) ** 2) ** 0.5
    thr = float(d.quantile(lote_quantile)) + pad_extra
    sel = lotes_df[d <= thr]
    if sel.empty:
        return None
    rects = sel["bbox"].tolist()
    x0 = min(bb[0] for bb in rects) - pad_inner
    y0 = min(bb[1] for bb in rects) - pad_inner
    x1 = max(bb[2] for bb in rects) + pad_inner
    y1 = max(bb[3] for bb in rects) + pad_inner
    return (x0, y0, x1, y1)

def build_quadra_regions(quadras_df, lotes_df, polys,
                         min_lotes_in_poly=DEFAULT_MIN_LOTES_IN_POLY,
                         lote_quantile=DEFAULT_FALLBACK_LOTE_QUANTILE,
                         pad_inner=DEFAULT_FALLBACK_PAD_INNER,
                         pad_extra=DEFAULT_FALLBACK_PAD_EXTRA):
    """
    Para cada QUADRA:
      - tenta polígono (maior área) e checa cobertura de LOTEs;
      - se pouca cobertura, usa CAIXA por proximidade dos LOTEs.
    Retorna uma lista de regiões heterogêneas:
      [{"label_text": "01", "kind":"poly", "geom": [(x,y),...]},
       {"label_text": "02", "kind":"box",  "geom": (x0,y0,x1,y1)}, ...]
    """
    regions = []
    if quadras_df is None or quadras_df.empty:
        return regions

    for _, q in quadras_df.iterrows():
        qlabel = q["text"]
        qx, qy = q["x"], q["y"]

        best_poly, lotes_in_poly = choose_best_poly_for_quadra(qx, qy, polys, lotes_df)

        if best_poly is not None and lotes_in_poly >= min_lotes_in_poly:
            regions.append({"label_text": qlabel, "kind": "poly", "geom": best_poly})
        else:
            box = make_fallback_box_for_quadra(qx, qy, lotes_df, lote_quantile, pad_inner, pad_extra)
            if box is not None:
                regions.append({"label_text": qlabel, "kind": "box", "geom": box})
            elif best_poly is not None:
                # sem LOTEs suficientes, mas não conseguimos caixa — use o poly mesmo
                regions.append({"label_text": qlabel, "kind": "poly", "geom": best_poly})
            # senão, fica sem região (improvável)

    return regions

def which_quadra_for_point_with_regions(x, y, regions):
    for r in regions:
        if r["kind"] == "poly":
            if point_in_poly(x, y, r["geom"]):
                return r["label_text"]
        else:  # box
            x0, y0, x1, y1 = r["geom"]
            if (x0 <= x <= x1) and (y0 <= y <= y1):
                return r["label_text"]
    return None

def auto_radius_if_needed(df_types, current_value):
    """Se current_value < 0, define como 3.5x altura mediana do texto disponível."""
    if current_value is not None and current_value >= 0:
        return current_value
    heights = []
    for df in df_types:
        if df is None or df.empty:
            continue
        heights.extend([(b[3] - b[1]) for b in df["bbox"]])
    if not heights:
        return 20.0  # fallback seguro
    med = pd.Series(heights).median()
    return 3.5 * float(med)

# =========================
# DEBUG VISUAL (opcional)
# =========================
def save_debug_regions_png(pdf_page, regions, quadras_df, out_path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("Aviso: matplotlib não disponível; ignorando debug visual.")
        return
    # Desenha todos paths para dar contexto
    polys = extract_polygons_from_drawings(pdf_page)
    fig, ax = plt.subplots(figsize=(9, 11))
    for p in polys:
        arr = pd.DataFrame(p["polygon"]).values
        ax.plot(arr[:, 0], arr[:, 1])
    # Regiões por quadra
    for r in regions:
        if r["kind"] == "poly":
            arr = pd.DataFrame(r["geom"]).values
            ax.plot(arr[:, 0], arr[:, 1], linewidth=2)
        else:
            x0, y0, x1, y1 = r["geom"]
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=2)
    # Rótulos de quadra
    if quadras_df is not None and not quadras_df.empty:
        for _, q in quadras_df.iterrows():
            ax.scatter([q["x"]], [q["y"]])
            ax.text(q["x"], q["y"], f"Q{q['text']}", fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title("Regiões por QUADRA (poly/box)")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Extrair LOTE / IMÓVEL / nº Residencial por QUADRA (vetorização).")
    parser.add_argument("pdf", type=str, help="Caminho do PDF de entrada")
    parser.add_argument("--out", type=str, default="resultado_by_quad.xlsx", help="Arquivo XLSX de saída")
    parser.add_argument("--max-imovel", type=float, default=DEFAULT_MAX_DIST_IMOVEL, help="Raio (px) para associar IMÓVEL ao LOTE (use negativo para auto)")
    parser.add_argument("--max-res", type=float, default=DEFAULT_MAX_DIST_RES, help="Raio (px) para associar nº Residencial ao LOTE (use negativo para auto)")
    parser.add_argument("--auto-radius", action="store_true", help="Usar raio automático (~3.5x altura mediana do texto)")
    parser.add_argument("--min-lotes-in-poly", type=int, default=DEFAULT_MIN_LOTES_IN_POLY, help="Mínimo de LOTEs dentro do polígono para aceitá-lo como região da QUADRA")
    parser.add_argument("--fallback-quantile", type=float, default=DEFAULT_FALLBACK_LOTE_QUANTILE, help="Quantil de distância para selecionar LOTEs na caixa de fallback")
    parser.add_argument("--fallback-pad-inner", type=float, default=DEFAULT_FALLBACK_PAD_INNER, help="Padding interno (px) da caixa de fallback")
    parser.add_argument("--fallback-pad-extra", type=float, default=DEFAULT_FALLBACK_PAD_EXTRA, help="Reforço (px) no raio para seleção de LOTEs na caixa de fallback")
    parser.add_argument("--debug-regions", action="store_true", help="Salvar PNG com regiões por página (poly/box)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_xlsx = Path(args.out)

    doc = fitz.open(str(pdf_path))
    all_rows = []

    for page_num, page in enumerate(doc, start=1):
        recs = extract_spans(page_num, page)
        df_raw = pd.DataFrame(recs)

        if df_raw.empty:
            continue

        quadras = df_raw[df_raw["type"] == "QUADRA"].copy()
        lotes   = df_raw[df_raw["type"] == "LOTE"].copy()
        imoveis = df_raw[df_raw["type"] == "IMOVEL"].copy()
        resids  = df_raw[df_raw["type"] == "RES"].copy()

        polys = extract_polygons_from_drawings(page)

        # construir regiões robustas por quadra
        regions = build_quadra_regions(
            quadras_df=quadras,
            lotes_df=lotes,
            polys=polys,
            min_lotes_in_poly=args.min_lotes_in_poly,
            lote_quantile=args.fallback_quantile,
            pad_inner=args.fallback_pad_inner,
            pad_extra=args.fallback_pad_extra
        )

        # debug visual opcional
        if args.debug_regions:
            dbg_png = out_xlsx.with_suffix(f".p{page_num}_regions.png")
            save_debug_regions_png(page, regions, quadras, dbg_png)

        # anotar quadra por região
        def tag_with_regions(df):
            if df.empty:
                df["quadra_key"] = None
                return df
            keys = []
            for _, r in df.iterrows():
                keys.append(which_quadra_for_point_with_regions(r["x"], r["y"], regions))
            df = df.copy()
            df["quadra_key"] = keys
            return df

        lotes   = tag_with_regions(lotes)
        imoveis = tag_with_regions(imoveis)
        resids  = tag_with_regions(resids)

        # raio (px): fixo ou automático
        MAXI = args.max_imovel
        MAXR = args.max_res
        if args.auto_radius:
            MAXI = auto_radius_if_needed([imoveis, lotes], -1)
            MAXR = auto_radius_if_needed([resids, lotes], -1)

        # associação por LOTE, restrita à mesma quadra
        rows = []
        for _, l in lotes.iterrows():
            qkey = l["quadra_key"]
            if qkey is None:
                continue

            # IMÓVEL (mesma quadra, dentro do raio)
            ip = imoveis[imoveis["quadra_key"] == qkey]
            imovel_val = None
            if not ip.empty:
                dists = ((ip["x"] - l["x"])**2 + (ip["y"] - l["y"])**2)**0.5
                min_idx = dists.idxmin()
                if dists.loc[min_idx] <= MAXI:
                    imovel_val = ip.loc[min_idx, "text"]

            # RES (mesma quadra, dentro do raio)
            rp = resids[resids["quadra_key"] == qkey]
            res_val = None
            if not rp.empty:
                d2 = ((rp["x"] - l["x"])**2 + (rp["y"] - l["y"])**2)**0.5
                min2 = d2.idxmin()
                if d2.loc[min2] <= MAXR:
                    res_val = rp.loc[min2, "text"]

            # manter apenas se tiver IMÓVEL (plus)
            if imovel_val is not None:
                rows.append({
                    "LOTE": l["text"],
                    "QUADRA": qkey,
                    "IMOVEL (plus)": imovel_val,
                    "nº Residencial": res_val
                })

        all_rows.extend(rows)

    # salvar
    df = pd.DataFrame(all_rows).drop_duplicates()
    df.to_excel(out_xlsx, index=False)
    print(f"✅ Planilha gerada: {out_xlsx}")

if __name__ == "__main__":
    main()

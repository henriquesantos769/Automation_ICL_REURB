
# -*- coding: utf-8 -*-
try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    import pymupdf as fitz

import argparse
import math
from pathlib import Path
import pandas as pd
import math
from collections import defaultdict
import re  # Importação adicionada para uso de expressões regulares

# =========================
# CONFIG PADRÃO
# =========================
DEFAULT_MAX_DIST_IMOVEL = -1    # -1 => auto
DEFAULT_MAX_DIST_RES    = -1

# ...
def auto_radius_if_needed(df_types, current_value):
    if current_value is not None and current_value >= 0:
        return current_value
    heights = []
    for df in df_types:
        if df is None or df.empty:
            continue
        heights.extend([(b[3] - b[1]) for b in df["bbox"]])
    if not heights:
        return 40.0               # fallback maior
    med = pd.Series(heights).median()
    return 6 * float(med)       # era 3.5x -> 4.5x

# Ajuste de cores mais próximo do seu PDF
COLOR_REF = {
    "QUADRA": (255,   0,   0),   # vermelho (nº grande com círculo)
    "LOTE":   (  0,   0, 255),   # azul
    "IMÓVEL": (  0, 180,   0),   # verde (um pouco menos saturado que (0,255,0))
    "RES":    (255, 165,   0),   # laranja
}

SNAP_TOL = 3.0
MIN_CYCLE_AREA = 600.0  
MAX_POLY_AREA = 2.5e6 

# tolerâncias um pouco mais permissivas
THRESH = {
    "QUADRA": 175.0,   # 160 -> 175
    "LOTE":   185.0,   # 170 -> 185
    "IMÓVEL": 220.0,   # 190 -> 205
    "RES":    205.0,   # 185 -> 205
}

def dist_point_to_bbox(px, py, bbox):
    """Menor distância Euclidiana do ponto (px,py) ao retângulo bbox=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = bbox
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    # clamping
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    return math.hypot(px - cx, py - cy)


def lot_distance(l_row, t_row, mode="bbox"):
    """
    Calcula distância entre um LOTE (linha 'l_row') e um alvo (linha 't_row').
    mode:
      - "bbox": distância ponto→bbox do LOTE (recomendado)
      - "center": distância centro→centro (comportamento antigo)
    """
    if mode == "bbox":
        return dist_point_to_bbox(t_row["x"], t_row["y"], l_row["bbox"])
    else:
        return math.hypot(t_row["x"] - l_row["x"], t_row["y"] - l_row["y"])


def associate_target_first(target_df, lotes_df, max_radius, dist_mode="bbox"):
    """
    Prioriza o ALVO (IMÓVEL/RES): cada alvo tenta o LOTE mais próximo
    na mesma quadra. Usa 'row_id' estável para evitar desalinhamento.
    Retorna {lote_row_id -> texto_do_alvo_ou_None}
    """
    out = {}
    if lotes_df is None or lotes_df.empty:
        return out

    # Index por quadra
    lotes_by_q = {}
    for _, l in lotes_df.iterrows():
        lotes_by_q.setdefault(str(l["quadra_key"]), []).append(int(l["row_id"]))

    targets_by_q = {}
    if target_df is not None and not target_df.empty:
        for _, t in target_df.iterrows():
            targets_by_q.setdefault(str(t["quadra_key"]), []).append(int(t["row_id"]))

    # lookup por row_id (rápido e imutável)
    lotes_lookup   = {int(l["row_id"]): l for _, l in lotes_df.iterrows()}
    targets_lookup = {int(t["row_id"]): t for _, t in target_df.iterrows()}

    for qkey, lote_ids in lotes_by_q.items():
        for lid in lote_ids:
            out[lid] = None

        tids = targets_by_q.get(qkey, [])
        if not tids:
            continue

        # preferências: para cada alvo, lista de (d, lid) em ordem crescente
        prefs = {}
        for tid in tids:
            trow = targets_lookup[tid]
            cand = []
            for lid in lote_ids:
                lrow = lotes_lookup[lid]
                d = lot_distance(lrow, trow, mode=dist_mode)
                if d <= max_radius:
                    cand.append((d, lid))
            cand.sort(key=lambda x: x[0])
            prefs[tid] = cand

        # Gale–Shapley simplificado: alvos propõem aos lotes
        free_t = set(tids)
        proposed_i = {tid: 0 for tid in tids}
        chosen = {}  # lid -> (tid, d)

        while free_t:
            tid = free_t.pop()
            lst = prefs.get(tid, [])
            i = proposed_i[tid]
            assigned = False
            while i < len(lst):
                d, lid = lst[i]
                proposed_i[tid] = i + 1
                if lid not in chosen:
                    chosen[lid] = (tid, d)  # lote livre
                    assigned = True
                    break
                else:
                    curr_tid, curr_d = chosen[lid]
                    if d < curr_d - 1e-9:     # fica o mais perto
                        chosen[lid] = (tid, d)
                        # o antigo tenta o próximo
                        if proposed_i[curr_tid] < len(prefs.get(curr_tid, [])):
                            free_t.add(curr_tid)
                        assigned = True
                        break
                i += 1
            # se não conseguiu nenhum, segue sem alocação

        # materializa para saída
        for lid, (tid, d) in chosen.items():
            out[lid] = targets_lookup[tid]["text"]

    return out


def associate_global_nearest(target_df, lotes_df, max_radius, dist_mode="bbox"):
    """
    Emparelhamento global guloso por quadra, usando distância ao LOTE.
    dist_mode: "bbox" (padrão) ou "center".
    Retorna: dict {lote_index -> texto_alvo_ou_None}
    """
    result = {}
    if lotes_df is None or lotes_df.empty:
        return result

    by_lotes = {}
    for lid, l in lotes_df.iterrows():
        by_lotes.setdefault(str(l["quadra_key"]), []).append(lid)

    by_targets = {}
    if target_df is not None and not target_df.empty:
        for tid, t in target_df.iterrows():
            by_targets.setdefault(str(t["quadra_key"]), []).append(tid)

    for qkey, lote_ids in by_lotes.items():
        tgt_ids = by_targets.get(qkey, [])
        for lid in lote_ids:
            result[lid] = None
        if not tgt_ids:
            continue

        pairs = []
        for lid in lote_ids:
            lrow = lotes_df.loc[lid]
            for tid in tgt_ids:
                trow = target_df.loc[tid]
                d = lot_distance(lrow, trow, mode=dist_mode)
                if d <= max_radius:
                    pairs.append((d, lid, tid))

        pairs.sort(key=lambda p: p[0])
        taken_l, taken_t = set(), set()
        for d, lid, tid in pairs:
            if (lid not in taken_l) and (tid not in taken_t):
                result[lid] = target_df.loc[tid, "text"]
                taken_l.add(lid)
                taken_t.add(tid)

    return result


def log_outline_widths(page):
    import numpy as np
    ws = [float(d.get("width",1.0)) for d in page.get_drawings() if _is_outline_stroke(d)]
    if ws:
        print("outline widths (min/med/max):", min(ws), np.median(ws), max(ws), "n=", len(ws))


def save_debug_segments_png(page, segs, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 11))
    for (x0,y0,x1,y1) in segs:
        ax.plot([x0,x1],[y0,y1], linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title("Todos os segmentos costurados (debug)")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def which_quadra_for_point_with_regions(x, y, regions):
    """
    Retorna o label da QUADRA cuja região (polígono) contém o ponto (x,y).
    Se nenhuma região contiver o ponto, retorna None.
    """
    if not regions:
        return None
    for r in regions:
        if r.get("kind") == "poly":
            if point_in_poly(x, y, r["geom"]):
                return r["label_text"]
    return None

def _angle(ax, ay, bx, by):
    """Ângulo de vetor A->B em rad (0..2π)."""
    ang = math.atan2(by - ay, bx - ax)
    return ang if ang >= 0 else (ang + 2*math.pi)

def _leftmost_next(prev_vec, candidates):
    """
    Dado vetor anterior (dx,dy) e vizinhos (lista de (nx,ny)), escolhe
    o vizinho com menor rotação no sentido anti-horário (leftmost).
    """
    px, py = prev_vec
    prev_ang = math.atan2(py, px)
    best = None
    best_rot = 1e9
    for (vx, vy, nid) in candidates:
        ang = math.atan2(vy, vx)
        rot = (ang - prev_ang) % (2*math.pi)
        if rot < best_rot:
            best_rot, best = rot, nid
    return best

# --- utils de cor (mantém) ---
def _rgb01_from_any(c):
    if c is None: return (0.0, 0.0, 0.0)
    if isinstance(c, (tuple, list)) and len(c) >= 3:
        r, g, b = float(c[0]), float(c[1]), float(c[2])
        if r > 1.0 or g > 1.0 or b > 1.0:  # veio 0..255
            r, g, b = r/255.0, g/255.0, b/255.0
        return (max(0,min(1,r)), max(0,min(1,g)), max(0,min(1,b)))
    if isinstance(c, int):
        c = c & 0xFFFFFF
        return ((c>>16)&255)/255.0, ((c>>8)&255)/255.0, (c&255)/255.0
    return (0.0, 0.0, 0.0)

# --- NOVO: heurística p/ traço de contorno preto/cinza ---
def _is_outline_stroke(d):
    r, g, b = _rgb01_from_any(d.get("color"))
    v = max(r, g, b)
    m = min(r, g, b)
    s = 0.0 if v == 0 else (v - m) / (v + 1e-9)

    # aceitar apenas cinza/preto (baixa saturação) e não muito claro
    if s > 0.30:        # corte de saturação (verde/azul/vermelho ficam de fora)
        return False
    if v > 0.90:        # cinza quase branco
        return False

 

    # NÃO cortar por largura: hairline (width==0) é comum nas plantas
    return True


# --- EXTRAÇÃO de segmentos só dos contornos ---
def extract_segments_from_drawings(page):
    segs = []
    for d in page.get_drawings():
        if not _is_outline_stroke(d):
            continue

        items = d.get("items", [])

        # retângulo direto
        r = d.get("rect")
        if r is not None and d.get("type") == "re":
            segs += [
                (r.x0, r.y0, r.x1, r.y0),
                (r.x1, r.y0, r.x1, r.y1),
                (r.x1, r.y1, r.x0, r.y1),
                (r.x0, r.y1, r.x0, r.y0),
            ]
            continue

        subpath = []

        def flush_path(close=False):
            nonlocal subpath
            for i in range(1, len(subpath)):
                x0, y0 = subpath[i-1]
                x1, y1 = subpath[i]
                segs.append((x0, y0, x1, y1))
            if close and len(subpath) >= 2:
                x0, y0 = subpath[-1]
                x1, y1 = subpath[0]
                segs.append((x0, y0, x1, y1))
            subpath = []

        for it in items:
            if not isinstance(it, tuple) or not it:
                continue
            cmd = it[0]
            if cmd == "m":
                flush_path(False)
                p = it[1]
                subpath = [(p.x, p.y)]
            elif cmd == "l":
                p1 = it[2]
                if not subpath:
                    p0 = it[1]
                    subpath = [(p0.x, p0.y)]
                subpath.append((p1.x, p1.y))
            elif cmd == "re":
                rect = it[1]
                segs += [
                    (rect.x0, rect.y0, rect.x1, rect.y0),
                    (rect.x1, rect.y0, rect.x1, rect.y1),
                    (rect.x1, rect.y1, rect.x0, rect.y1),
                    (rect.x0, rect.y1, rect.x0, rect.y0),
                ]
            elif cmd == "h":
                flush_path(True)
            # se existirem curvas 'c' no seu PDF e quiser achatar:
            # elif cmd == "c":
            #     # TODO: discretizar curvas se necessário

        if d.get("closePath", False):
            flush_path(True)
        else:
            flush_path(False)
    return segs

# (Opcional) para entender as cores que o PDF usa:
def dump_drawing_palette(page, top=20):
    from collections import Counter
    cols = []
    for d in page.get_drawings():
        cols.append(_rgb01_from_any(d.get("color")))
    cnt = Counter(cols)
    print("Top cores de traço:", cnt.most_common(top))


# ====== NOVO: snap de vértices (aglomeração por grade) ======
def snap_points(points, tol=SNAP_TOL):
    """
    Agrupa pontos (x,y) em centros a <= tol de distância.
    Implementação simples O(N^2) porém robusta (não gera None).
    Retorna:
      centers: [(cx, cy), ...]
      index_map: para cada ponto original, o índice do centro correspondente
    """
    centers = []     # lista de (cx, cy)
    counts  = []     # quantos pontos foram agregados ao centro
    index_map = []

    for (x, y) in points:
        best_i = -1
        best_d = 1e18
        for i, (cx, cy) in enumerate(centers):
            d = math.hypot(cx - x, cy - y)
            if d < best_d:
                best_d = d
                best_i = i

        if best_i != -1 and best_d <= tol:
            # atualiza o centro via média incremental
            cnt = counts[best_i] + 1
            cx, cy = centers[best_i]
            centers[best_i] = ((cx * counts[best_i] + x) / cnt,
                               (cy * counts[best_i] + y) / cnt)
            counts[best_i] = cnt
            index_map.append(best_i)
        else:
            centers.append((x, y))
            counts.append(1)
            index_map.append(len(centers) - 1)

    return centers, index_map

# ====== NOVO: grafo e “polygonização” por caminhada leftmost ======
def polygonize_segments(segs, tol=SNAP_TOL, min_area=MIN_CYCLE_AREA):
    """
    segs: [(x0,y0,x1,y1), ...]
    Fecha ciclos a partir dos segmentos 'soldados' (snap) e retorna polígonos.
    """

    if not segs:
        return []

    # 1) junta todos os endpoints e faz o SNAP (solda vértices próximos)
    endpoints = []
    for (x0, y0, x1, y1) in segs:
        endpoints.append((x0, y0))
        endpoints.append((x1, y1))
    centers, idx_map = snap_points(endpoints, tol=tol)  # centers = lista de nós “soldados”

    # 2) cria arestas não orientadas entre os nós "snapped"
    edges = set()
    n_cent = len(centers)
    for i in range(len(segs)):
        a = idx_map[2*i]      # índice do nó do (x0,y0)
        b = idx_map[2*i + 1]  # índice do nó do (x1,y1)
        if a == b:
            continue
        if not (0 <= a < n_cent and 0 <= b < n_cent):
            continue
        e = (min(a, b), max(a, b))
        edges.add(e)

    # 3) grafo de adjacência
    from collections import defaultdict
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # 4) half-edges e varredura "leftmost" para fechar ciclos
    half_edges = set()
    for a, b in edges:
        half_edges.add((a, b))
        half_edges.add((b, a))

    visited = set()
    polys = []

    for he in list(half_edges):
        if he in visited:
            continue
        start = he
        path = [start[0], start[1]]
        visited.add(start)
        u, v = start

        # vetor direção anterior: v - u
        pvx = centers[v][0] - centers[u][0]
        pvy = centers[v][1] - centers[u][1]

        while True:
            # candidatos a partir de v
            cand = []
            for w in adj[v]:
                if w == u:
                    continue
                if not (0 <= w < len(centers)) or not (0 <= v < len(centers)):
                    continue
                vx = centers[w][0] - centers[v][0]
                vy = centers[w][1] - centers[v][1]
                cand.append((vx, vy, w))
            if not cand:
                break  # beco sem saída

            w = _leftmost_next((pvx, pvy), cand)
            if (v, w) in visited:
                break

            visited.add((v, w))
            path.append(w)
            u, v = v, w
            pvx = centers[v][0] - centers[u][0]
            pvy = centers[v][1] - centers[u][1]

            # ciclo fechado?
            if w == path[0] and len(path) > 3:
                poly = [centers[i] for i in path[:-1]]
                A = poly_area(poly)
                if A >= min_area and A <= MAX_POLY_AREA:
                    polys.append(poly)
                break

    return polys


# ====== SUBSTITUI: escolha do polígono da QUADRA via costura ======
def build_regions_polys_stitched(quadras_df, lotes_df, page,
                                 min_lotes_in_poly=6,
                                 snap_tol=SNAP_TOL,
                                 min_cycle_area=MIN_CYCLE_AREA):
    """
    Gera regiões por QUADRA usando:
      1) costura de segmentos -> polygonize
      2) escolhe o polígono que contém o rótulo e maximiza #LOTES internos
    """
    regions = []
    if quadras_df is None or quadras_df.empty:
        return regions

    segs = extract_segments_from_drawings(page)
    stitched = polygonize_segments(segs, tol=snap_tol, min_area=min_cycle_area)

    for _, q in quadras_df.iterrows():
        qx, qy = q["x"], q["y"]
        best_poly, best_score = None, -1
        for poly in stitched:
            if not point_in_poly(qx, qy, poly):
                continue
            # quantos LOTEs ficam dentro
            cnt = 0
            if lotes_df is not None and not lotes_df.empty:
                for _, l in lotes_df.iterrows():
                    if point_in_poly(l["x"], l["y"], poly):
                        cnt += 1
            if cnt > best_score:
                best_score = cnt
                best_poly = poly
        if best_poly is not None and best_score >= min_lotes_in_poly:
            regions.append({"label_text": q["text"], "kind": "poly", "geom": best_poly})

    return regions

# ====== NOVO: debug com legenda por tipo ======
def save_debug_regions_png_with_legend(pdf_page, regions, quadras, lotes, imoveis, resids, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 11))

    # desenha regiões escolhidas
    for r in regions:
        poly = r["geom"]
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=2, label=f"Q{r['label_text']}")

    # pontos por tipo
    lg = []
    if lotes is not None and not lotes.empty:
        ax.scatter(lotes["x"], lotes["y"], marker="s", s=12, label="LOTE")
    if imoveis is not None and not imoveis.empty:
        ax.scatter(imoveis["x"], imoveis["y"], marker="^", s=16, label="IMÓVEL (plus)")
    if resids is not None and not resids.empty:
        ax.scatter(resids["x"], resids["y"], marker="o", s=14, label="nº Residencial")
    if quadras is not None and not quadras.empty:
        for _, q in quadras.iterrows():
            ax.scatter([q["x"]], [q["y"]], marker="*", s=60, color="red")
            ax.text(q["x"], q["y"], f"Q{q['text']}", fontsize=9)

    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title("Regiões por QUADRA (polígonos costurados)")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================
# UTILS
# =========================
def int_to_rgb(val):
    """
    Converte qualquer formato de cor do PyMuPDF para (R,G,B) 0..255.
    Aceita: int 0xRRGGBB (positivo ou negativo), tuple/list 0..1, tuple/list 0..255.
    """
    if isinstance(val, (tuple, list)) and len(val) >= 3:
        r, g, b = val[0], val[1], val[2]
        # floats 0..1
        if isinstance(r, float) or isinstance(g, float) or isinstance(b, float):
            r = int(round(max(0.0, min(1.0, float(r))) * 255))
            g = int(round(max(0.0, min(1.0, float(g))) * 255))
            b = int(round(max(0.0, min(1.0, float(b))) * 255))
            return r, g, b
        # inteiros 0..255
        return int(r) & 255, int(g) & 255, int(b) & 255

    if isinstance(val, int):
        val = val & 0xFFFFFF
        r = (val >> 16) & 255
        g = (val >> 8) & 255
        b = val & 255
        return r, g, b

    return 0, 0, 0



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
    inside = False
    n = len(poly)
    if n < 3:
        return False
    x0, y0 = poly[0]
    for i in range(1, n+1):
        x1, y1 = poly[i % n]
        cond = ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
        if cond:
            inside = not inside
        x0, y0 = x1, y1
    return inside

def poly_area(poly):
    if not poly or len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0

# =========================
# VETORES → POLÍGONOS (fechamento robusto)
# =========================
def extract_polygons_from_drawings(page, eps=1e-3):
    """
    Extrai polígonos fechados da página, tratando subpaths e fechamento via 'h'
    e/ou flag closePath. Continua capturando 're' (retângulos).
    """
    polys = []
    drawings = page.get_drawings()
    for d in drawings:
        items = d.get("items", [])
        stroke = d.get("color")
        width  = d.get("width", 1.0)
        fill   = d.get("fill")

        # retângulo direto
        r = d.get("rect")
        if r is not None and d.get("type") == "re":
            poly = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
            polys.append({"polygon": poly, "stroke": stroke, "width": width, "fill": fill})
            continue

        subpath = []
        subpath_start = None

        def flush_subpath(force=False):
            # fecha se: (force) ou (primeiro≈último)
            nonlocal subpath, subpath_start
            if len(subpath) >= 3:
                x0, y0 = subpath[0]
                xn, yn = subpath[-1]
                if force or math.hypot(xn - x0, yn - y0) < eps:
                    polys.append({"polygon": subpath[:], "stroke": stroke, "width": width, "fill": fill})
            subpath = []
            subpath_start = None

        for it in items:
            if not isinstance(it, tuple) or not it:
                continue
            cmd = it[0]

            if cmd == 'm':  # moveTo → inicia novo subpath
                # descarrega o anterior, se existir
                flush_subpath(force=False)
                p = it[1]
                subpath = [(p.x, p.y)]
                subpath_start = (p.x, p.y)

            elif cmd == 'l':  # lineTo
                # 'l' vem como ('l', p0, p1)
                p1 = it[2]
                if not subpath:
                    # se não houve 'm' antes, inicie com p0
                    p0 = it[1]
                    subpath = [(p0.x, p0.y)]
                subpath.append((p1.x, p1.y))

            elif cmd == 're':  # rect como item
                rect = it[1]
                poly = [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)]
                polys.append({"polygon": poly, "stroke": stroke, "width": width, "fill": fill})

            elif cmd == 'h':  # closePath explícito
                flush_subpath(force=True)

            # (curvas 'c' ignoradas, pois não costumam definir os limites dos lotes)

        # closePath no nível do drawing
        if d.get("closePath", False):
            flush_subpath(force=True)
        else:
            flush_subpath(force=False)

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
                if not any(ch.isdigit() for ch in text):
                    continue
                
                # NOVO: Filtra para aceitar apenas strings numéricas (ou com pouca letra)
                # O seu código já tinha uma boa tentativa, mas isso garante um filtro mais forte.
                digits_only = re.sub(r'[^0-9]', '', text)
                if not digits_only:
                    continue

                r, g, b = int_to_rgb(s["color"])
                typ = classify_by_nearest_color((r, g, b))

                size = float(s.get("size", 0))
                if typ == "QUADRA" and size < 9.0:
                    typ = "OTHER"

                if typ == "OTHER":
                    continue

                # NOVO: filtro específico para o IMÓVEL (plus)
                if typ == "IMÓVEL" and not (re.fullmatch(r'\d{5}', text)):
                    continue
                    
                x0, y0, x1, y1 = s["bbox"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                
                # Ajusta a lógica para pegar o texto original em vez de `digits_only`
                # exceto para LOTE e RES que podem ter letras no final
                if typ == "LOTE" or typ == "RES":
                    value = digits_only
                else:
                    value = text

                recs.append({
                    "page": page_num,
                    "text": value,
                    "type": typ,
                    "x": cx, "y": cy,
                    "bbox": (x0, y0, x1, y1),
                    "size": size,
                })
    return recs


# =========================
# PARTIÇÃO POR QUADRA (sem caixas sobrepostas)
# =========================
def assign_nearest_quadra(df_points, quadras):
    """Para cada ponto (LOTE/IMÓVEL/RES), atribui a quadra do rótulo de QUADRA mais próximo."""
    if df_points is None or df_points.empty:
        return df_points
    if quadras is None or quadras.empty:
        df_points["quadra_key"] = None
        return df_points

    qx = quadras["x"].values
    qy = quadras["y"].values
    qlabels = quadras["text"].values

    keys = []
    for _, r in df_points.iterrows():
        dx = qx - r["x"]
        dy = qy - r["y"]
        d2 = dx*dx + dy*dy
        j = int(d2.argmin())
        keys.append(str(qlabels[j]))
    out = df_points.copy()
    out["quadra_key"] = keys
    return out

# =========================
# RAIO AUTOMÁTICO
# =========================


# =========================
# ASSOCIAÇÃO EXCLUSIVA POR LOTE
# =========================
def associate_unique(target_df, lotes_df, max_radius):
    """
    target_df: pontos (IMÓVEL ou RES) já com 'quadra_key'
    lotes_df : LOTEs já com 'quadra_key'
    Retorna dict: lote_index -> texto_alvo (ou None)
    (cada ponto é usado no máximo uma vez por quadra)
    """
    taken = set()
    result = {}

    if lotes_df.empty:
        return result

    by_q = {}
    for idx, row in target_df.iterrows():
        by_q.setdefault(row["quadra_key"], []).append((idx, row["x"], row["y"], row["text"]))

    for lid, l in lotes_df.iterrows():
        q = l["quadra_key"]
        best_idx = None
        best_d = 1e18
        if q in by_q:
            for idx, tx, ty, tval in by_q[q]:
                if idx in taken:
                    continue
                d = math.hypot(tx - l["x"], ty - l["y"])
                if d < best_d:
                    best_d, best_idx = d, idx
        if best_idx is not None and best_d <= max_radius:
            result[lid] = target_df.loc[best_idx, "text"]
            taken.add(best_idx)
        else:
            result[lid] = None
    return result

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Extrair LOTE / IMÓVEL (plus) / nº Residencial por QUADRA.")
    parser.add_argument("pdf", type=str, help="Caminho do PDF de entrada")
    parser.add_argument("--out", type=str, default="resultado_by_quad.xlsx", help="Arquivo XLSX de saída")
    parser.add_argument("--max-imovel", type=float, default=DEFAULT_MAX_DIST_IMOVEL,
                        help="Raio (px) para associar IMÓVEL ao LOTE (use negativo para auto)")
    parser.add_argument("--max-res", type=float, default=DEFAULT_MAX_DIST_RES,
                        help="Raio (px) para associar nº Residencial ao LOTE (use negativo para auto)")
    parser.add_argument("--auto-radius", action="store_true",
                        help="Usar raio automático (~3.5x altura mediana do texto)")
    # ---- novos controles do modo poligonal ----
    parser.add_argument("--min-lotes-in-poly", type=int, default=6,
                        help="Mín. de LOTEs dentro do polígono para aceitar como região da QUADRA")
    parser.add_argument("--snap-tol", type=float, default=SNAP_TOL,
                        help="Tolerância (px) para 'soldar' vértices na costura de segmentos")
    parser.add_argument("--min-cycle-area", type=float, default=MIN_CYCLE_AREA,
                        help="Área mínima (px²) para aceitar um ciclo como polígono")
    parser.add_argument("--debug-regions", action="store_true",
                        help="Salvar PNG com regiões por página (polígonos + pontos)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    
    out_xlsx = Path(args.out)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)


    doc = fitz.open(str(pdf_path))
    all_rows = []

    for page_num, page in enumerate(doc, start=1):
        segs = extract_segments_from_drawings(page)
        if args.debug_regions:
            dbg_segs = out_xlsx.with_name(out_xlsx.stem + f".p{page_num}_segments.png")
            save_debug_segments_png(page, segs, dbg_segs)

        # log preliminar de segmentos
        print(f"[p{page_num}] segmentos extraídos: {len(segs)}")
        recs = extract_spans(page_num, page)
        df_raw = pd.DataFrame(recs)

        
        if df_raw.empty:
            print(f"[p{page_num}] nenhum span colorido detectado (df_raw vazio).")
            # ainda assim, tente mostrar polígonos costurados (pode ajudar no tuning)
            regions = build_regions_polys_stitched(
                quadras_df=pd.DataFrame(columns=["x","y","text"]),  # vazio
                lotes_df=pd.DataFrame(columns=["x","y","text"]),    # vazio
                page=page,
                min_lotes_in_poly=args.min_lotes_in_poly,
                snap_tol=args.snap_tol,
                min_cycle_area=args.min_cycle_area,
            )
            if args.debug_regions:
                dbg_png = out_xlsx.with_name(out_xlsx.stem + f".p{page_num}_polyregions.png")
                save_debug_regions_png_with_legend(page, regions,
                                                pd.DataFrame(), pd.DataFrame(),
                                                pd.DataFrame(), pd.DataFrame(),
                                                dbg_png)
            continue

        quadras = df_raw[df_raw["type"] == "QUADRA"].copy()
        lotes   = df_raw[df_raw["type"] == "LOTE"].copy()
        imoveis = df_raw[df_raw["type"] == "IMÓVEL"].copy()
        resids  = df_raw[df_raw["type"] == "RES"].copy()

        print(f"[p{page_num}] spans={len(df_raw)} | Q={len(quadras)} L={len(lotes)} I={len(imoveis)} R={len(resids)}")
        if quadras.empty or lotes.empty:
            print(f"[p{page_num}] sem QUADRA ou LOTE → pulando página")
            continue # sem quadra ou sem lote não dá para montar linhas

        # ---------- NOVO: construir regiões poligonais por QUADRA ----------
        regions = build_regions_polys_stitched(
            quadras_df=quadras,
            lotes_df=lotes,
            page=page,
            min_lotes_in_poly=args.min_lotes_in_poly,
            snap_tol=args.snap_tol,
            min_cycle_area=args.min_cycle_area,
        )

        def tag_with_regions(df_points):
            if df_points.empty:
                df_points["quadra_key"] = None
                return df_points

            keys = []
            for _, r in df_points.iterrows():
                # 1) usar região somente se contiver o ponto
                qkey = which_quadra_for_point_with_regions(r["x"], r["y"], regions)

                # 2) se não couber em nenhuma região, usar SEMPRE o rótulo vermelho mais próximo
                if qkey is None:
                    d2 = (quadras["x"] - r["x"])**2 + (quadras["y"] - r["y"])**2
                    qkey = str(quadras.loc[d2.idxmin(), "text"])

                keys.append(str(qkey))
            out = df_points.copy()
            out["quadra_key"] = keys
            return out


        # ---------- usar regiões + fallback ----------
        lotes   = tag_with_regions(lotes)
        imoveis = tag_with_regions(imoveis)
        resids  = tag_with_regions(resids)

        for df in (lotes, imoveis, resids):
            df.reset_index(drop=True, inplace=True)
            df["row_id"] = df.index

        # ---------- raio fixo ou automático ----------
        MAXI = auto_radius_if_needed([imoveis, lotes], args.max_imovel)
        MAXR = auto_radius_if_needed([resids,  lotes], args.max_res)
        '''
        # DEBUG: logar top-3 lotes mais próximos para cada IMÓVEL
        for tid, t in imoveis.iterrows():
            cand = []
            same_q = lotes[lotes["quadra_key"] == t["quadra_key"]]
            for lid, l in same_q.iterrows():
                d = lot_distance(l, t, mode="bbox")
                cand.append((d, lid, l["text"]))
            cand.sort(key=lambda x: x[0])
            print(f"[debug] IMOVEL {t['text']} @({t['x']:.1f},{t['y']:.1f}) -> {[(round(d,1), lt) for d,_,lt in cand[:3]]}")
        '''
        # ---------- associação EXCLUSIVA por LOTE (dentro da mesma quadra) ----------
        imovel_map = associate_target_first(imoveis, lotes, MAXI, dist_mode="bbox")
        resid_map  = associate_target_first(resids,  lotes, MAXR, dist_mode="bbox")
        '''
        # (opcional) auditoria com row_id
        rev_imovel = {v:k for k,v in imovel_map.items() if v is not None}
        for _, t in imoveis.iterrows():
            lid = rev_imovel.get(t["text"])
            msg = f"[audit] IMOVEL {t['text']} -> "
            msg += f"LOTE {lotes.loc[lotes['row_id']==lid, 'text'].values[0]}" if lid is not None else "sem lote"
            print(msg)
        '''
        # gerar linhas: SEM usar o index do pandas
        for _, l in lotes.iterrows():
            lid = int(l["row_id"])
            imovel_val = imovel_map.get(lid)
            if imovel_val is None:
                continue
            res_val = resid_map.get(lid)
            all_rows.append({
                "LOTE": l["text"],
                "QUADRA": l["quadra_key"],
                "IMÓVEL (plus)": imovel_val,
                "nº Residencial": res_val
            })
        '''
        print(f"[p{page_num}] spans={len(df_raw)} | Q={len(quadras)} L={len(lotes)} I={len(imoveis)} R={len(resids)}")
        print(f"[p{page_num}] raioI={MAXI:.1f} raioR={MAXR:.1f}")
        print("[drawings]", len(page.get_drawings()))
        '''
        dump_drawing_palette(page)
        log_outline_widths(page)
        segs = extract_segments_from_drawings(page)
        
        '''
        print(">> DUPLICATAS DE LOTE NA MESMA QUADRA")
        dup = lotes.groupby(["quadra_key","text"]).size()
        print(dup[dup>1])
        '''

        # ---------- debug opcional ----------
        if args.debug_regions:
            
            dbg_segs = out_xlsx.with_name(out_xlsx.stem + f".p{page_num}_segments.png")
            save_debug_segments_png(page, segs, dbg_segs)

            dbg_png = out_xlsx.with_name(out_xlsx.stem + f".p{page_num}_polyregions.png")
            save_debug_regions_png_with_legend(page, regions, quadras, lotes, imoveis, resids, dbg_png)

        
        if regions:
            lotes   = lotes[lotes["quadra_key"].notna()].copy()
            imoveis = imoveis[imoveis["quadra_key"].notna()].copy()
            resids  = resids[resids["quadra_key"].notna()].copy()
            print("Distribuição de LOTE por quadra:", lotes["quadra_key"].value_counts().to_dict())


    df = pd.DataFrame(all_rows).drop_duplicates()
    df.to_excel(out_xlsx, index=False)
    print(f"[OK] Planilha gerada: {out_xlsx}")


if __name__ == "__main__":
    main()
"""Funciones compartidas por varias fases del cuaderno VC_P4_RTDETR.

- Detección con RT-DETR (Fases 0, 4, 5, 6 y 8): detect, cargar_detector_matriculas.
- Vídeo (Fases 5 y 8): a_numpy, a_detections, assign_plate_to_car.
- OCR con Florence-2 (Fases 6, 7 y 8): recorte, preparación, lectura por lotes y extracción de la matrícula.
- Datos y evaluación del OCR (Fases 6 y 7): recoger, eval_ocr_v2, resumen, por_alto.

Las funciones no leen variables globales del cuaderno: el modelo, el procesador y las rutas se pasan como
argumentos. Los únicos valores por defecto compartidos son MARGEN y PREP_MODE (ver abajo).

Las funciones que usan el repositorio de RT-DETR (src.core) lo importan dentro de la función, así que antes
hay que añadir RT-DETR/rtdetrv2_pytorch al sys.path (lo hace la celda de configuración de cada fase).
"""
import re
import json
import time
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


MARGEN    = 0.25     # margen alrededor de la caja, proporcional a su ancho/alto
PREP_MODE = "pad"    # "pad": relleno gris hasta un cuadrado | "estirar": se lleva a 2:1 antes de rellenar


# ======================================================================================================
# Detección con RT-DETR
# ======================================================================================================
@torch.no_grad()
def detect(model, postprocessor, bgr, size, conf=0.6):
    """Detección con RT-DETR sobre una imagen BGR (OpenCV).
    size: lado de la entrada cuadrada; debe coincidir con el eval_spatial_size del modelo
          (640 para los modelos COCO, 800 para el de matrículas).
    Devuelve (cajas xyxy en píxeles de la imagen original, scores, etiquetas) con score >= conf."""
    device = next(model.parameters()).device
    h, w = bgr.shape[:2]
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None].to(device)
    labels, boxes, scores = postprocessor(model(x), torch.tensor([[w, h]], device=device))
    labels, boxes, scores = labels[0].cpu().numpy(), boxes[0].cpu().numpy(), scores[0].cpu().numpy()
    keep = scores >= conf
    return boxes[keep], scores[keep], labels[keep]


def cargar_detector_matriculas(cfg_path, ckpt_path, device):
    """Carga el RT-DETRv2 entrenado para matrículas (pesos EMA de best.pth) en modo deploy.
    Se descartan del checkpoint los buffers 'anchors' y 'valid_mask' (dependen del tamaño de entrada y se
    regeneran al construir el modelo); cargarlos tal cual daba error, por eso strict=False.
    Devuelve (modelo, postprocesador, época del checkpoint)."""
    from src.core import YAMLConfig                 # repo de RT-DETR (tiene que estar en sys.path)
    from src.core.yaml_utils import load_config
    load_config.__defaults__[0].clear()             # evita que se mezclen configs YAML cargadas antes
    cfg = YAMLConfig(str(cfg_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)   # incluye el estado del optimizador
    state = {k: v for k, v in ckpt["ema"]["module"].items() if not k.endswith(("anchors", "valid_mask"))}
    cfg.model.load_state_dict(state, strict=False)
    epoca = ckpt.get("last_epoch")
    del ckpt
    return cfg.model.deploy().to(device), cfg.postprocessor.deploy().to(device), epoca


# ======================================================================================================
# Vídeo: supervision y asignación matrícula -> vehículo
# ======================================================================================================
def a_numpy(x):
    """Tensor de torch -> array de numpy (supervision trabaja con numpy)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def a_detections(boxes, scores, labels):
    """Salida de detect() -> sv.Detections (vacío si no hay cajas)."""
    import supervision as sv
    if len(boxes) == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=a_numpy(boxes).astype(np.float32),
                         confidence=a_numpy(scores).astype(np.float32),
                         class_id=a_numpy(labels).astype(int))


def fraction_plate(box_plate, box_car):
    """Fracción del área de la matrícula que cae dentro del vehículo. Cajas xyxy en píxeles."""
    x1, y1 = max(box_plate[0], box_car[0]), max(box_plate[1], box_car[1])
    x2, y2 = min(box_plate[2], box_car[2]), min(box_plate[3], box_car[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_plate = max(0.0, box_plate[2] - box_plate[0]) * max(0.0, box_plate[3] - box_plate[1])
    return inter / area_plate if area_plate > 0 else 0.0


def plate_in_car(box_plate, box_car, umbral=0.9):
    """La matrícula pertenece al vehículo si al menos el 90 % de su área cae dentro (puede sobresalir un poco)."""
    return fraction_plate(box_plate, box_car) >= umbral


def assign_plate_to_car(boxes_plate, boxes_car, umbral=0.9):
    """Para cada matrícula, índice del vehículo que la contiene (-1 si ninguno).
    Si la contienen varios (vehículos solapados) se elige el de menor área: el más cercano a la matrícula."""
    asignaciones = []
    for bp in boxes_plate:
        mejor, mejor_area = -1, np.inf
        for j, bc in enumerate(boxes_car):
            area = (bc[2] - bc[0]) * (bc[3] - bc[1])
            if plate_in_car(bp, bc, umbral) and area < mejor_area:
                mejor, mejor_area = j, area
        asignaciones.append(mejor)
    return asignaciones


# ======================================================================================================
# OCR con Florence-2
# ======================================================================================================
# Formato actual español: 4 dígitos + 3 consonantes (sin vocales, Ñ ni Q)
LETRAS_OK = "BCDFGHJKLMNPRSTVWXYZ"
PLATE_RE  = re.compile(rf"\d{{4}}[{LETRAS_OK}]{{3}}")
# Confusiones típicas del OCR según la posición (las 4 primeras son dígitos, las 3 últimas letras)
TO_DIGIT  = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "T": "1",
             "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "D", "O": "D", "Q": "D", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "U": "V"}


def to_square(bgr, fill=114):
    """Rellena el recorte con gris hasta un cuadrado (sin deformarlo) y lo pasa a PIL RGB.
    Florence-2 redimensiona siempre a 768x768 y una matrícula es muy apaisada (~4.7:1)."""
    h, w = bgr.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(bgr, top, s - h - top, left, s - w - left,
                            cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))


def extraer_matricula(texto):
    """Devuelve la matrícula 1234BCD contenida en el texto leído, o '' si no hay ninguna creíble."""
    t = re.sub(r"[^A-Z0-9]", "", str(texto).upper())
    m = PLATE_RE.search(t)                    # 1) tal cual (p. ej. 'E1685MBY' -> '1685MBY')
    if m:
        return m.group()
    mejor, n_mejor = "", 99                   # 2) ventana de 7 con el menor nº de correcciones por posición
    for i in range(len(t) - 6):
        w = t[i:i + 7]
        c = "".join(TO_DIGIT.get(x, x) for x in w[:4]) + "".join(TO_LETTER.get(x, x) for x in w[4:])
        n = sum(a != b for a, b in zip(w, c))
        if PLATE_RE.fullmatch(c) and n < n_mejor:
            mejor, n_mejor = c, n
    return mejor if n_mejor <= 2 else ""      # más de 2 correcciones: lectura poco fiable, se descarta


def crop_plate_v2(bgr, box, margen=None):
    """Recorta la caja xyxy con un margen proporcional (MARGEN por defecto). None si queda degenerada."""
    margen = MARGEN if margen is None else margen
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
    if x2 - x1 < 4 or y2 - y1 < 2:
        return None
    return bgr[y1:y2, x1:x2]


def preparar(crop, modo=None):
    """Recorte BGR -> PIL cuadrado para Florence-2 (PREP_MODE por defecto)."""
    modo = PREP_MODE if modo is None else modo
    if modo == "estirar":
        h, w = crop.shape[:2]
        if w > 2 * h:
            crop = cv2.resize(crop, (w, w // 2), interpolation=cv2.INTER_CUBIC)
    return to_square(crop)


@torch.no_grad()
def read_plates_v2(crops_bgr, model, processor, batch_size=16, num_beams=3, modo=None):
    """OCR por lotes con la tarea <OCR> de Florence-2.
    Devuelve [(texto_crudo, matricula_o_vacio, confianza_o_None), ...]. La confianza es la probabilidad
    media por token de la secuencia (solo existe con num_beams > 1)."""
    param = next(model.parameters())
    salida = []
    for i in range(0, len(crops_bgr), batch_size):
        imgs = [preparar(c, modo) for c in crops_bgr[i:i + batch_size]]
        inputs = processor(text=["<OCR>"] * len(imgs), images=imgs, return_tensors="pt").to(param.device, param.dtype)
        gen = model.generate(**inputs, max_new_tokens=20, num_beams=num_beams, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
        textos = processor.batch_decode(gen.sequences, skip_special_tokens=True)
        ss = getattr(gen, "sequences_scores", None)       # log-prob media por token (solo con beams)
        confs = ss.float().exp().cpu().tolist() if ss is not None else [None] * len(textos)
        salida += [(t, extraer_matricula(t), c) for t, c in zip(textos, confs)]
    return salida


# ======================================================================================================
# Datos para el OCR: recortes de las matrículas del dataset
# ======================================================================================================
def abrir_para_ocr(info, img_dir, orig_dirs=(), usar_original=True):
    """Imagen de la que se recorta para el OCR: la original girada por EXIF (como en la Fase 1) si existe en
    alguna carpeta de orig_dirs; si no, la del dataset. Devuelve (bgr, escala_x, escala_y, es_original);
    las cajas COCO se multiplican por la escala."""
    nombre = info.get("orig_name")
    if usar_original and nombre:
        for d in orig_dirs:
            p = Path(d) / nombre
            if p.is_file():
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                sx, sy = im.size[0] / info["width"], im.size[1] / info["height"]
                if sx >= 1.0 and abs(sx - sy) <= 0.02 * sx:          # misma proporción: las cajas cuadran
                    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), sx, sy, True
    return cv2.imread(str(Path(img_dir) / info["file_name"])), 1.0, 1.0, False


def recoger(split, dataset_coco, orig_dirs=(), usar_original=True, alto_max=64, ctx_x=0.4, ctx_y=0.6):
    """Matrículas legibles de formato actual de un split, con contexto alrededor (para poder variar el margen).
    Cada muestra: ctx (recorte BGR con contexto), box (caja de la matrícula dentro de ctx), text,
    alto (alto en el dataset a 1280 px, para comparar tablas), alto_px (alto real en ctx), original, file.
    Se cachea en <dataset_coco>/ocr_recortes_<split>.pkl (se regenera si cambian los parámetros)."""
    dataset_coco = Path(dataset_coco)
    cache = dataset_coco / f"ocr_recortes_{split}.pkl"
    clave = dict(ctx_x=ctx_x, ctx_y=ctx_y, alto_max=alto_max, original=usar_original,
                 dirs=[str(d) for d in orig_dirs])
    if cache.is_file():
        datos = pickle.loads(cache.read_bytes())
        if datos["clave"] == clave:
            ms = [dict(m, ctx=cv2.imdecode(np.frombuffer(m["ctx"], np.uint8), cv2.IMREAD_COLOR)) for m in datos["muestras"]]
            print(f"{split}: {len(ms)} recortes desde caché ({sum(m['original'] for m in ms)} de fotos originales)")
            return ms

    d = json.loads((dataset_coco / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
    img_dir = dataset_coco / f"images_{split}"
    info = {i["id"]: i for i in d["images"]}
    por_img = defaultdict(list)
    for a in d["annotations"]:
        if a.get("legible") and PLATE_RE.fullmatch(a.get("text") or ""):
            por_img[a["image_id"]].append(a)
    muestras, t0 = [], time.time()
    for img_id, anns in por_img.items():
        bgr, sx, sy, es_orig = abrir_para_ocr(info[img_id], img_dir, orig_dirs, usar_original)
        H, W = bgr.shape[:2]
        for a in anns:
            x, y, w, h = a["bbox"]
            x, w, y, h = x * sx, w * sx, y * sy, h * sy               # caja en la imagen de la que se recorta
            x1, y1 = int(max(0, x - ctx_x * w)), int(max(0, y - ctx_y * h))
            x2, y2 = int(min(W, x + w + ctx_x * w)), int(min(H, y + h + ctx_y * h))
            ctx, box = bgr[y1:y2, x1:x2], (x - x1, y - y1, x + w - x1, y + h - y1)
            f = min(1.0, alto_max / h)                                # limita RAM sin perder nada útil
            if f < 1.0:
                ctx = cv2.resize(ctx, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                box = tuple(v * f for v in box)
            muestras.append({"ctx": ctx.copy(), "box": box, "text": a["text"], "alto": a["bbox"][3],
                             "alto_px": h * f, "alto_orig": h, "original": es_orig,
                             "file": info[img_id]["file_name"]})
        del bgr
    cache.write_bytes(pickle.dumps({"clave": clave, "muestras": [
        dict(m, ctx=cv2.imencode(".png", m["ctx"])[1].tobytes()) for m in muestras]}))
    print(f"{split}: {len(muestras)} recortes en {time.time() - t0:.0f} s "
          f"({sum(m['original'] for m in muestras)} de fotos originales) -> {cache.name}")
    return muestras


# ======================================================================================================
# Evaluación del OCR
# ======================================================================================================
def resumen(filas, titulo=""):
    """Imprime exact match, % de lecturas aceptadas (dan una matrícula válida) y precisión de las aceptadas.
    Devuelve el exact match."""
    exact = np.mean([f["gt"] == f["pred"] for f in filas])
    acept = [f for f in filas if f["pred"]]
    prec  = np.mean([f["gt"] == f["pred"] for f in acept]) if acept else 0.0
    print(f"{titulo:<28} exact {100 * exact:5.1f} % | aceptadas {100 * len(acept) / len(filas):5.1f} % "
          f"| precisión aceptadas {100 * prec:5.1f} %")
    return exact


def eval_ocr_v2(muestras, model, processor, margen=None, modo=None, num_beams=3):
    """OCR sobre muestras = recoger(...), recortando cada matrícula de su contexto con la caja REAL."""
    crops, meta = [], []
    for m in muestras:
        c = crop_plate_v2(m["ctx"], m["box"], margen)
        if c is not None:
            crops.append(c); meta.append(m)
    preds = read_plates_v2(crops, model, processor, modo=modo, num_beams=num_beams)
    return [dict(file=m["file"], gt=m["text"], alto=m["alto"], alto_px=m["alto_px"], original=m["original"],
                 crudo=r, pred=p, conf=c) for m, (r, p, c) in zip(meta, preds)]


def por_alto(filas, bins=(0, 10, 14, 18, 24, 32, 1e9), clave="alto"):
    """Tabla por alto de la matrícula. clave='alto': alto a 1280 px (comparable entre tablas);
    'alto_px': alto real del recorte."""
    print(f"\n{clave + ' (px)':>10} | {'n':>4} | {'exact':>7} | {'precisión aceptadas':>20}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = [f for f in filas if lo <= f[clave] < hi]
        if not g:
            continue
        a = [f for f in g if f["pred"]]
        pa = f"{100 * np.mean([f['gt'] == f['pred'] for f in a]):6.1f} %" if a else "-"
        print(f"{f'{lo:.0f}-{hi:.0f}' if hi < 1e9 else f'>{lo:.0f}':>10} | {len(g):>4} | "
              f"{100 * np.mean([f['gt'] == f['pred'] for f in g]):5.1f} % | {pa:>20}")


"""
Fichero que contiene funciones que se repiten
"""

import cv2
import os
from pathlib import Path
from collections import Counter, deque
import numpy as np

# ========== Funciones para ejecutar el filtro del Hamster ==========
HAMSTER_DIR = Path("hamster")
EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Emotes en orden de PRIORIDAD: si en un frame se cumplen varios, gana el primero de la lista.
# Cada nombre coincide con su carpeta dentro de hamster/
EMOTES = ["thinking", "meh", "bah", "actually", "thumbs up", "thumbs down", "workout", "kiss"]

# --- Umbrales ajustables ---
UMBRAL_BESO = 0.60       # Blendshape "mouthPucker" (morritos)
VENTANA_EMOTE = 6        # Nº de frames para la votación por mayoría (evita parpadeos)
EXTENDIDO = 1.15         # Un dedo está extendido si dist(muñeca, punta) > EXTENDIDO * dist(muñeca, PIP)

# Índices de los landmarks de la mano (MediaPipe)
MUNECA, PULGAR_MCP, PULGAR_IP, PULGAR_TIP = 0, 2, 3, 4
INDICE_MCP, INDICE_PIP, INDICE_TIP = 5, 6, 8
CORAZON_MCP = 9
PIPS = {"indice": 6, "corazon": 10, "anular": 14, "menique": 18}
TIPS = {"indice": 8, "corazon": 12, "anular": 16, "menique": 20}

CONEXIONES_MANO = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                   (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


# ---------------------------------------------------------------- Imágenes de hámster
def leer_imagen(path):
    """cv2.imread falla con rutas no ASCII en Windows (p. ej. '☝️🤓.png'), por eso se usa imdecode."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def cargar_hamsters(nombres):
    """Devuelve {nombre: [imágenes]} leyendo hamster/<nombre>/ para cada emoción o emote."""
    hamsters = {}
    for nombre in nombres:
        carpeta = HAMSTER_DIR / nombre
        imgs = [leer_imagen(f) for f in carpeta.iterdir() if f.suffix.lower() in EXTS] if carpeta.is_dir() else []
        hamsters[nombre] = [im for im in imgs if im is not None]
        if not hamsters[nombre]:
            print(f"Aviso: no hay imágenes para '{nombre}' en {carpeta}")
    return hamsters


def pegar_imagen(fondo, overlay, x1, y1, x2, y2):
    """Redimensiona overlay al tamaño de la caja y lo pega sobre fondo (respetando la transparencia)."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    overlay = cv2.resize(overlay, (w, h))
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    roi = fondo[y1:y2, x1:x2]
    if overlay.shape[2] == 4:  # PNG con canal alfa
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        roi[:] = (alpha * overlay[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    else:
        roi[:] = overlay


# ---------------------------------------------------------------- Geometría de la mano
def mano_a_pixeles(mano, w, h):
    """Landmarks normalizados de MediaPipe -> array (21, 2) en píxeles."""
    return np.array([[lm.x * w, lm.y * h] for lm in mano], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def estado_dedos(p):
    """Para una mano en píxeles devuelve qué dedos están extendidos y el tamaño de la palma.
    Se compara la distancia muñeca-punta con muñeca-PIP, que no depende de la rotación de la mano."""
    palma = dist(p[MUNECA], p[CORAZON_MCP])
    d = {dedo: dist(p[MUNECA], p[TIPS[dedo]]) > EXTENDIDO * dist(p[MUNECA], p[PIPS[dedo]]) for dedo in TIPS}
    # Pulgar extendido: su punta se aleja del nudillo del índice y queda más lejos de la muñeca que su IP
    d["pulgar"] = (dist(p[PULGAR_TIP], p[INDICE_MCP]) > 0.65 * palma
                   and dist(p[MUNECA], p[PULGAR_TIP]) > dist(p[MUNECA], p[PULGAR_IP]))
    d["palma"] = palma
    d["centro"] = p.mean(axis=0)
    return d


def tres_doblados(d):
    """Corazón, anular y meñique doblados."""
    return not (d["corazon"] or d["anular"] or d["menique"])


def es_vertical(v, factor=1.3):
    return abs(v[1]) > factor * abs(v[0])


# ---------------------------------------------------------------- Detección de emotes
def emotes_manos(manos, caja):
    """Devuelve el conjunto de emotes de mano que se cumplen. manos = lista de arrays (21, 2) en píxeles,
    caja = (x1, y1, x2, y2) de la cara detectada por YOLO (sirve de referencia de posición y escala)."""
    x1, y1, x2, y2 = caja
    fw, fh = x2 - x1, y2 - y1
    fcy = y1 + fh / 2
    info = [estado_dedos(p) for p in manos]
    emotes = set()

    # --- Gestos con dos manos ---
    if len(manos) == 2:
        (p1, d1), (p2, d2) = zip(manos, info)
        palma = (d1["palma"] + d2["palma"]) / 2
        # thinking: manos juntas/cruzadas (centros muy cerca) o yemas de los índices tocándose
        if dist(d1["centro"], d2["centro"]) < 1.6 * palma or dist(p1[INDICE_TIP], p2[INDICE_TIP]) < 0.35 * palma:
            emotes.add("thinking")
        # meh: las dos palmas abiertas, una a cada lado de la cara y por debajo de los ojos
        abiertas = all(sum(d[k] for k in TIPS) >= 3 for d in (d1, d2))
        izq, der = sorted((d1["centro"], d2["centro"]), key=lambda c: c[0])
        if abiertas and izq[0] < x1 and der[0] > x2 and izq[1] > fcy and der[1] > fcy:
            emotes.add("meh")

    # --- Gestos con una mano ---
    for p, d in zip(manos, info):
        tip = p[INDICE_TIP]
        if tres_doblados(d):
            # bah: el índice es la yema más alta y está junto a la boca/barbilla
            indice_arriba = tip[1] <= min(p[TIPS[k]][1] for k in TIPS)
            if (indice_arriba and x1 - 0.1 * fw <= tip[0] <= x2 + 0.1 * fw
                    and y1 + 0.45 * fh <= tip[1] <= y2 + 0.15 * fh):
                emotes.add("bah")
            # actually: índice extendido apuntando hacia arriba
            v = tip - p[INDICE_MCP]
            if d["indice"] and v[1] < 0 and es_vertical(v, 1.5):
                emotes.add("actually")

        if not d["indice"] and tres_doblados(d):  # Puño (con o sin pulgar)
            v = p[PULGAR_TIP] - p[PULGAR_MCP]
            nudillos_y = [p[PIPS[k]][1] for k in PIPS]
            if d["pulgar"] and es_vertical(v):
                if v[1] < 0 and p[PULGAR_TIP][1] < min(nudillos_y):
                    emotes.add("thumbs up")
                elif v[1] > 0 and p[PULGAR_TIP][1] > max(nudillos_y):
                    emotes.add("thumbs down")
            # workout: puño cerrado (pulgar recogido) a la altura de la cabeza y a un lado de la cara (bíceps)
            cx, cy = d["centro"]
            if not d["pulgar"] and cy < y2 and (cx < x1 - 0.2 * fw or cx > x2 + 0.2 * fw):
                emotes.add("workout")
    return emotes


def blendshapes_dict(resultado_cara):
    """Resultado del FaceLandmarker -> {nombre_blendshape: score} de la primera cara ({} si no hay)."""
    if resultado_cara is None or not resultado_cara.face_blendshapes:
        return {}
    return {c.category_name: c.score for c in resultado_cara.face_blendshapes[0]}


def emotes_cara(blend):
    """Emotes faciales: beso (blendshape mouthPucker)."""
    emotes = set()
    if blend.get("mouthPucker", 0) > UMBRAL_BESO:
        emotes.add("kiss")
    return emotes


def elegir_emote(candidatos):
    """Devuelve el emote de mayor prioridad (orden de EMOTES) o None."""
    return next((e for e in EMOTES if e in candidatos), None)


class Suavizador:
    """Votación por mayoría en los últimos n frames: el emote solo se activa si aparece en más de la
    mitad de ellos. Evita que la imagen parpadee por detecciones sueltas."""
    def __init__(self, n=VENTANA_EMOTE):
        self.hist = deque(maxlen=n)

    def __call__(self, valor):
        self.hist.append(valor)
        top, cnt = Counter(self.hist).most_common(1)[0]
        return top if cnt > self.hist.maxlen // 2 else None


# ---------------------------------------------------------------- Dibujo (depuración)
def dibujar_mano(img, p, color=(0, 255, 0)):
    for a, b in CONEXIONES_MANO:
        cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), color, 1)
    for x, y in p.astype(int):
        cv2.circle(img, (x, y), 3, color, -1)


def dibujar_cara(img, resultado_cara, w, h, color=(255, 255, 0)):
    """Dibuja los landmarks de la malla facial de MediaPipe (un punto por landmark)."""
    if resultado_cara is None or not resultado_cara.face_landmarks:
        return
    for lm in resultado_cara.face_landmarks[0]:
        cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 1, color, -1)


def dibujar_texto(img, texto, pos, color=(0, 255, 0), escala=0.6):
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4)   # Borde negro para leerlo mejor
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2)









# ========== Funciones usadas en el eigenfaces ==========

def LoadDataset(folder, ext):
    # Contador de número de clases del conjunto
    nclasses = 0
    # Contador de muestras por clase
    nperclass = []
    # Etiqueta de cada clase (nombre de la subcarpeta)
    classlabels = []
    # Inicializa estructuras de datos y sus correpondientes etiquetas
    X = []
    Y = []
   
    # Asume que en la ruta indicada hay una subcarpeta por clase
    for class_name in os.listdir(folder):
        # Cada subcarpeta implica una clase más
        nclasses += 1
        # Inicialmente esta clase no tiene muestras
        nsamples = 0    

        # Compone la ruta
        class_folder = os.path.join(folder, class_name)
        for file_name in os.listdir(class_folder):
            # Imágenes en formato ext
            if file_name.endswith(ext):
                # Lee la imagen
                image = cv2.imread (os.path.join(class_folder, file_name))

                # Lugar para aplicar un RoI sobre la imagen de entrada
                # Ejemplo para la mitad superior
                # image = image[0:int(height/2),:]

                # Extrae tamaños
                height, width, depth = image.shape
                # Convierte  grises
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Añade a X, previa conversión en vector
                X.append(gray.reshape(height*width))

                # Añade etiqueta numérica de la muestra
                Y.append(nclasses-1)

                #Incrementa el número de muestras
                nsamples += 1
        
        nperclass.append(nsamples)
        classlabels.append(class_name)
    
    return X, Y, nsamples, class_name, nperclass, classlabels, width, height

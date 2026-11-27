# Práctica 5. Biometría: Detección y caracterización de caras

## Práctica desarrollada por (Autoría):

- Juan Francisco Almeida Santana, [Github](https://github.com/kikoalmtana)
- Texenery Bordón Rodríguez, [Github](https://github.com/texem4k)

## Contenidos

- **Descripción de la práctica** 
- **Filtro con detector de emociones y gestos básicos** 
  - **Comparativa SVM+PCA, Yolo11s-cls y deepface**
- **Filtro con tracking de la nariz**
- **Uso de IA**

## Descripción  

La práctica 5 tiene como objetivo crear dos filtros o escenarios de aplicación usando información extraída del rostro mediante modelos de detección de caras y uno de ellos debe utilizar un modelo entrenado para la detección de información biométrica (por ejemplo, emociones).

En esta práctica se han desarrollado 2 filtros usando las tecnologías descritas en el guión de la práctica 5 y en prácticas anteriores los siguientes filtros:

- Un filtro que detecta emociones y en función a ellas muestra una imagen de un Hamster acorde a la emoción o gesto detectado
- Un filtro que trackeando la posición de la nariz te permite jugar una imitación del Flappy Bird

En este documento se muestra el desarrollo y se explica el código elaborado para ambos filtros


## 1. Filtro con detección de emociones y gestos básicos

El filtro coloca sobre la cara una imagen de un hámster que cambia según la emoción que se pone o el gesto que se hace con las manos. Cada vez que cambia la emoción o el gesto suena además un audio del hámster. El código está en [`emotion_detector.ipynb`](./emotion_detector.ipynb) y las funciones auxiliares (gestos, pegado de imágenes, sonidos) en [`utils.py`](./utils.py).

En cada frame de la webcam se usan **tres modelos**:

1. **YOLO11s** (Fine-tuning con WIDER FACE) localiza la cara.
2. **MediaPipe** (`HandLandmarker` y `FaceLandmarker`) obtiene los puntos de las manos y los *blendshapes* de la cara para reconocer gestos.
3. **YOLO11s-cls** (Fine-tuning con FER-2013) clasifica la emoción del recorte de la cara.

Si se reconoce un gesto, tiene prioridad sobre la emoción. Si no, se usa la emoción más probable del clasificador. La etiqueta elegida se elige al azar una imagen de `hamster/<etiqueta>/` y se pega sobre la caja de la cara. Solo se cambia de imagen cuando cambia la etiqueta.

### 1.1 Detector de caras: YOLO11s con WIDER FACE

El dataset de partida tiene **12.880 imágenes** con sus etiquetas en formato YOLO (una sola clase, `face`). La preparación fue:

- División aleatoria con semilla fija en **train / val / test = 70 / 15 / 15** (9.016 / 1.932 / 1.932 imágenes).
- Reescalado a **640x640 con *letterbox*** (se mantiene la relación de aspecto y se rellena con gris), ajustando las cajas a la nueva geometría.
- Limpieza de etiquetas: se recortan las cajas a los límites de la imagen y se descartan las de área nula (27 cajas), que Ultralytics marcaría como corruptas.

Se entrenó desde `yolo11s.pt` durante **100 épocas** (batch 8, AMP, sin volteo vertical porque no aporta en caras). Duró **11 h 47 min** en una RTX 3060 Laptop, y la mejor época fue la **82**.

| | Validación (época 82) | Test (`best.pt`) |
|---|---|---|
| Precisión | 0.855 | 0.851 |
| Recall | 0.654 | 0.603 |
| F1 | 0.741 | 0.706 |
| mAP50 | 0.733 | 0.679 |
| mAP50-95 | 0.390 | 0.359 |

<img src="./readme_assets/detector_caras_entrenamiento.png" alt="Curvas de entrenamiento del detector de caras" width="100%">

La precisión es alta: cuando detecta una cara, casi siempre lo es. El recall es la principal limitación, porque se escapan las caras muy pequeñas o tapadas de las escenas con mucha gente. Para este filtro no es un problema, ya que con la cara grande y de frente de una webcam las confianzas están en 0.8-0.9.

<img src="./readme_assets/detector_caras_test.png" alt="Ejemplos de detecciones en test" width="100%">

### 1.2 Clasificador de emociones: YOLO11s-cls con FER-2013

FER-2013 tiene caras en gris de **48x48** con 7 emociones: *angry, disgust, fear, happy, neutral, sad, surprise*. Hay **28.709 imágenes de train** y **7.178 de test**. Está muy desbalanceado: *disgust* tiene 436 imágenes de train, frente a las 7.215 de *happy*.

Se entrenó `yolo11s-cls.pt`, con estos ajustes:

- **50 épocas**, batch 64, `imgsz=48` (Ultralytics lo sube a 64, el múltiplo de 32 más cercano), AMP y paciencia de 10 épocas.
- *Data augmentation* por defecto de clasificación (volteo horizontal, RandAugment y *random erasing*) sin volteo vertical.
- Optimizador elegido automáticamente por Ultralytics (MuSGD).
- Duración: **47 min**. La mejor época fue la **48**, con una *accuracy top-1* de **0.689** y top-5 de 0.993.

<img src="./readme_assets/yolo11s_cls_entrenamiento.png" alt="Curvas de entrenamiento de YOLO11s-cls" width="60%">

La pérdida de validación se estanca a partir de la época 30 mientras la de train sigue bajando, así que con más épocas empezaría a sobreajustar.

`dataset_emociones` no tiene carpeta `val`, así que se usó **test como validación** para elegir `best.pt`.

### 1.3 Gestos (emotes)

Los gestos se detectan con reglas geométricas sobre los 21 puntos de cada mano de MediaPipe, usando la caja de la cara como referencia de posición y escala. Un dedo cuenta como extendido si la distancia muñeca-punta supera 1,15 veces la distancia muñeca-nudillo central (PIP). Esta comparación no depende de la rotación de la mano.

| Gesto | Cómo se detecta |
|---|---|
| `thinking` | Dos manos juntas o cruzadas, o las yemas de los índices tocándose |
| `meh` | Dos palmas abiertas, una a cada lado de la cara y por debajo de los ojos |
| `bah` | Índice levantado junto a la boca o la barbilla, con el resto de dedos doblados |
| `actually` | Índice extendido apuntando hacia arriba |
| `thumbs up` / `thumbs down` | Puño con el pulgar en vertical, por encima o por debajo de los nudillos |
| `workout` | Puño cerrado a la altura de la cabeza y a un lado de la cara (bíceps) |
| `kiss` | *Blendshape* `mouthPucker` > 0.60 del `FaceLandmarker` (morritos) |

Si se cumplen varios gestos a la vez, gana el primero de la tabla. Para evitar parpadeos, un gesto solo se activa si aparece en más de la mitad de los últimos 6 frames **(votación por mayoría)**.

### 1.4 Sonidos y controles

Con cada cambio de etiqueta suena un audio aleatorio de `hamster_sounds/<etiqueta>/`. Entre dos sonidos pasan al menos 8 s y cada uno se corta a los 6 s. El sonido empieza silenciado.

- `e`: mostrar u ocultar el hámster
- `r`: mostrar u ocultar los *landmarks* de MediaPipe
- `s`: activar o silenciar el sonido
- `ESC`: salir

### 1.5 Comparativa SVM+PCA, YOLO11s-cls y DeepFace

Para valorar el clasificador de emociones se compara con otros dos:

- **Eigenfaces (PCA) + SVM**, el método clásico visto en clase, entrenado por nosotros en [`VC_P5_eigenfaces_emociones.ipynb`](./VC_P5_eigenfaces_emociones.ipynb).
- **DeepFace**, con su modelo de emociones preentrenado (`DeepFace.analyze(actions=['emotion'])`).

La comparativa está en [`VC_P5_deepface_analyze.ipynb`](./VC_P5_deepface_analyze.ipynb).

#### Entrenamiento de Eigenfaces + SVM

Con el fin de simplificar, se ha reutilizado gran parte del cuaderno original `VC_P5_eigenfaces_handcrafted_kfold.ipynb` pero realizando ajustes para adaptarlo a un clasificador de emociones.

**1) K-fold de descriptores.** Primero se compararon varios descriptores con un *5-fold* estratificado. Con un SVM con kernel `RBF` y `GridSearch` (tal y cómo estaba originalmente), el dataset completo tardaría horas, así que se usó un submuestreo de **500 imágenes por clase** (3.436 en total, porque *disgust* solo tiene 436). El SVM usa `class_weight='balanced'` y un GridSearch con `C ∈ {1e3, 5e3}` y `gamma ∈ {1e-4 … 1e-2}`.

| Descriptor + clasificador | Precisión (macro) | Recall (macro) |
|---|---|---|
| Píxeles + KNN (k=5) | 0.234 | 0.235 |
| PCA + KNN | 0.230 | 0.236 |
| **PCA + SVM** | **0.288** | **0.292** |
| HOG (4x4) + SVM | 0.341 | 0.345 |
| LBP (3x3, `nri_uniform`) + SVM | 0.336 | 0.339 |
| Stacking (PCA + HOG + LBP) | 0.386 | 0.408 |

El PCA del 95 % de la varianza se queda con unas 216 componentes de las 2.304 que tiene cada imagen. Ninguna combinación pasa del 41 %. Los descriptores locales (HOG, LBP) funcionan mejor que PCA porque describen bordes y texturas de zonas concretas de la cara, mientras que PCA trabaja con la imagen entera.

**2) Modelo final.** Se entrenó un único pipeline **PCA (95 %) → MinMaxScaler → SVC RBF** con **todo el train (28.709 imágenes)**. Se reutilizaron los hiperparámetros que eligió el GridSearch para PCA+SVM en 4 de los 5 folds (`C=1000`, `gamma=0.0001`). El modelo resultante tiene **257 componentes** y **27.467 vectores soporte**, es decir, el 96 % de las muestras de train. Eso indica que el SVM apenas consigue separar las clases en el espacio de eigenfaces. El modelo se guarda en `models/eigenfaces_svm.joblib`.

<img src="./readme_assets/eigenfaces.png" alt="Cara media y primeras eigenfaces" width="100%">

Las primeras eigenfaces recogen sobre todo la iluminación (la 1 explica el 28,9 % de la varianza), la pose y la forma de la cara, no la expresión.

#### Evaluación en test

Los tres modelos clasifican las mismas **7.178 imágenes de `dataset_emociones/test`**. En DeepFace se usa `detector_backend='skip'`, porque las caras ya vienen recortadas, así que solo se compara el clasificador.

| | YOLO11s-cls | Eigenfaces + SVM | DeepFace |
|---|---|---|---|
| Accuracy | **0.688** | 0.333 | 0.564 |
| Precisión (macro) | **0.687** | 0.308 | 0.568 |
| Recall (macro) | **0.664** | 0.352 | 0.534 |
| F1 (macro) | **0.672** | 0.301 | 0.547 |
| ms / imagen | **0.5** (GPU, lotes de 256) | 6.5 (CPU) | 12.7 (GPU) |

F1 por emoción:

| | YOLO11s-cls | Eigenfaces + SVM | DeepFace |
|---|---|---|---|
| angry | **0.622** | 0.204 | 0.462 |
| disgust | **0.680** | 0.118 | 0.521 |
| fear | **0.511** | 0.173 | 0.408 |
| happy | **0.877** | 0.506 | 0.769 |
| neutral | **0.647** | 0.328 | 0.508 |
| sad | **0.569** | 0.301 | 0.422 |
| surprise | **0.795** | 0.478 | 0.741 |

<img src="./readme_assets/comparativa_confusion.png" alt="Matrices de confusión de los tres clasificadores" width="100%">

<img src="./readme_assets/comparativa_f1.png" alt="F1 por emoción de los tres clasificadores" width="80%">

**Conclusiones**

- **YOLO11s-cls es el mejor en todas las emociones** y también el más rápido, porque clasifica por lotes en GPU. Su resultado está algo inflado porque se usó test para elegir la mejor época, aunque la diferencia con los demás es demasiado grande para explicarse solo por eso.
- **DeepFace queda en una posición intermedia (≈56 %)**, lo esperable para la CNN pequeña de FER-2013 que usa. Comete los mismos errores que YOLO: *happy* y *surprise* son fáciles, mientras que *fear*, *sad* y *angry* se confunden entre sí. Su modelo también se entrenó con FER-2013, así que su cifra en test puede estar algo inflada.
- **Eigenfaces + SVM (≈33 %) está muy por debajo**. PCA compara píxeles en bruto y capta sobre todo la iluminación, la pose y la identidad, no los pequeños cambios de boca, cejas y ojos que definen una emoción. Además, `class_weight='balanced'` hace que prediga *disgust* demasiadas veces (recall 0.54, pero precisión 0.07).



## 2. Filtro con tracking de la nariz

## Datasets usados

- [WIDER FACE for YOLO (Para detector de caras)](https://www.kaggle.com/datasets/lylmsc/wider-face-for-yolo-training)
- [Hámster, (recogido a mano)](https://drive.google.com/file/d/145PJbvVxZh81FnwlnNA7bkNlE9ihglpk/view?usp=sharing)
- [FER-2013 (Para clasificador de emociones)](https://www.kaggle.com/code/shivambhardwaj0101/emotion-detection-fer-2013/notebook)

## Referencias

- [Detección y predicción en Ultralytics](https://docs.ultralytics.com/es/tasks/detect)
- [Como usar un clasificador de Ultralytics](https://docs.ultralytics.com/es/tasks/classify)

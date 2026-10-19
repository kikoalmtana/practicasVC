# Práctica 2. Funciones básicas de OpenCV

## Práctica desarrollada por:

- Juan Francisco Almeida Santana, [Github](https://github.com/texem4k)
- Texenery Bordón Rodríguez, [Github](https://github.com/texem4k)
## Contenidos

- **Descripción** 
- **Planteamiento de cada tarea**
- **Conversaciones con IA**

## Descripción  

El objetivo de esta práctica es aprender a extraer características geométricas y visuales de objetos presentes en imágenes para clasificarlos de forma autmática sin usar modelos de aprendizaje profundo.

Las tareas realizadas han sido las siguientes:

- Determinar la cantidad de dinero y monedas presentes en una imagen captada con una webcam

- Desarrollar un método de detección y clasificación de partículas de Alquitrán, Pellet y Fragment en una imagen a raíz de características geométricas 

## Planteamiento de cada tarea
  
### Tarea 1: parte de texe



### Tarea 2: Clasificación de partículas sin aprendizaje profundo

Esta tarea consiste en crear un clasificador de partículas de distintos tipos (Alquitran, Fragment y Pellet) para una imagen sin usar aprendizaje profundo:

![Imagen entrenamiento](./MPs_test.jpg)

La tarea principalmente tiene 2 partes:

1) La primera consiste en desarrollar un método de segmentación y detección de partículas en una imagen
2) La segunda consiste en desarrollar un método de clasificación en base a características encontradas en la imagen

En este documento se explica de forma resumida los métodos utilizados y el procedimiento que se llevó a cabo para completar esta tarea, los detalles y explicaciones detalladas acerca de los análisis de las características y de las decisiónes de los umbrales elegidos para la clasificación se encuentran en el cuaderno de la tarea junto al código ejecutable, donde se realizan comparaciones y se muestran tablas con datos de interés para la tarea.

### Método de segmentación y detección

Para encontrar las partículas presentes en una imagen se han desarrollado diferentes métodos que combinándolos, son capaces de extraer las partículas presentes en una imagen, para ello nos centramos en la detección de contornos en la imagen y para facilitar la detección se aplican máscaras binarias y distintos filtros:

- Método "segmentar": es un método implementado que dada una imagen y ciertos parámetros de configuracion (ajustados mediante prueba y error), aplica una máscara binaria adaptativa en una imagen en escala de grises (ADAPTIVE_THRESH_GAUSSIAN_C).

- Método "extraer_partículas": es un método implementado que recibe la máscara binaria que devuelve el método "segmentar", que se encarga de detectar los contornos en esta máscara y luego extraer las características del contorno, como su posición y otros valores interesantes.

- Método "características_contorno": es un método implementado y que se usa en el método "extraer_partículas" que dado un contorno, devuelve sus características.

Combinando todos estos métodos mas alguno mas implementado de visualización (que no interesa en la explicación de la extracción de partículas) se obtienen estos resultados:

![Ejemplo extracción de partículas](./tarea2_img/imagenes_readme/ejemplo1_tarea1.png)

Con esto ya podríamos segmentar partículas en una imagen y extraer características de las mismas.

### Método de clasificación en base a umbrales

Dada la limitación de no poder usar aprendizaje profundo para detectar y clasificar, se ha optado por definir diferentes umbrales en base a las características físicas de las partículas en las imagenes para su clasificación. Estos umbrales se han definido en torno a las siguientes características:

- **Circularidad de la partícula:** La circularidad define como es de circular la forma de una particula, de 0 (nada) a 1 (círculo perfecto). Para ello se ha estudiado los valores de circularidad de las partículas en sus distintas imágenes de entrenamiento.

![Circularidad Alquitran](./tarea2_img/imagenes_readme/ejemplo1_tarea2.png)
![Circularidad Fragment](./tarea2_img/imagenes_readme/ejemplo2_tarea2.png)
![Circularidad Pellet](./tarea2_img/imagenes_readme/ejemplo3_tarea2.png)

En estas gráficas se observan los diferentes resultados en el estudio de la circularidad de las distintas poblaciones de partículas en las imagenes de entrenamiento, a simple vista se puede ver que esta característica nos sirve para diferenciar Fragment de Alquitran y Pellet (porque tienen a una circularidad mas variada y que tiende a ser menor) y que las "más circulares" con diferencia son las partículas de Fragment.

- **Color de la partícula en Lab**: El modelo de color Lab es un modelo de color pensado para describir todos los colores que el ojo humano puede ver, esto lo hace utilizando un componente de "Luminosidad" (L) y dos ejes que miden el croma de la partícula (a* y b*).

![Circularidad Alquitran](./tarea2_img/imagenes_readme/ejemplo4_tarea2.png)
![Circularidad Fragment](./tarea2_img/imagenes_readme/ejemplo5_tarea2.png)
![Circularidad Pellet](./tarea2_img/imagenes_readme/ejemplo6_tarea2.png)

Observando las diferencias en esta representación del color entre las partículas podemos apreciar que especialmente las partículas de alquitran tienden a tener valores mas bajos de L valores de a* y b* cercanos a 0, al contrario que las de Pellet y Fragment.

Estudiando estas dos características en las imagenes de ejemplo podemos establecer umbrales que podemos aplicar en cada particula detectada tras segmentar la imagen de prueba para clasificarla entre las 3 clases.

### Resultados

![Imagen de test segmentada](./tarea2_img/imagenes_readme/imagen_test_segmentada.png)

Tras procesar la imagen de test utilizando este método de segmentación descrito podemos apreciar ruido en la segmentación y detección de particulas, debido a irregularidades en el color de fondo de la imagen original, además de partículas que se quedan "fuera de la máscara".

De las 120 partículas presentes en la imagen de test, se detectan 127, pero sólo se logran emparejar correctamente 62, lo que da un acierto de poco mas del 50% en la detección de partículas.

![Matriz de confusión final](./tarea2_img/imagenes_readme/matriz%20de%20confusiónpng.png)

Por otro lado en la clasificación utilizando umbrales se ha logrado un 75% de acierto en la clasificación de las partículas detectadas correctamente en la segmentación.

## Conversaciones con IA

- Conversaciones con Claude, 
    - Finalidad -> Obtener y desarrollar un primer planteamiento inicial usando un modelo de Aprendizaje Automático (RandomForestClassifier), descartado posteriormente
    - [Enlace](https://claude.ai/share/ef804d83-414a-40dc-a802-4e8c6affd69c)

    - Finalidad -> Desarrollo del concepto actual de clasificador (basado en umbrales)
    - [Enlace](https://claude.ai/share/8da05385-e3bd-4a30-b957-1d9790a536f4)


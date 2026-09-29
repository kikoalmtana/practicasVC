# Práctica 2. Funciones básicas de OpenCV

## Contenidos

- **Descripción** 
- **Planteamiento de cada tarea**
- **Conversaciones con IA**
- **Referencias**

## Descripción  

Al igual que en la práctica anterior, se han llevado a cabo una serie de ejercicios básicos para una toma de contacto con la librería OpenCV, así cómo acceder y modificar el valor de un pixel, umbralizados y entre otros. Las tareas llevadas a cabo son las siguientes:

- Realizar la cuenta de píxeles blancos por fila. Determinar el valor máximo de píxeles blancos para filas, mostrando el nº de filas y sus respectivas posiciones para aquellas filas que cuyo cantidad de balnc pixeles blanco sean mayores al 90% del valor máximo obtenido. Resaltar estas filas.

- Aplicar el umbralizado de la imagen resultante de Sobel (8 bits) y realizar el conteo de filas y columnas parecido al del ejercicio anterior. Destacar aquellas filas y columnas cuyos porcentaje de blanco supere el 90% del máximo. Remarcar las filas y columnas que cumplan dicha condición. Comparar resultados entre Sobel y Canny.

- Tras ver los vídeos planteados, proponer una demostración de la parte de procesamiento de la imagen, tomando como punto de partida alguna de los videos

## Planteamiento de cada tarea
  
### Tarea 1, Detección de filas con mayor cantidad de píxeles (Canny)

Siguiendo el planteamiento de la celda anterior a la tarea ()

Al ser una imagen de 800x800, cada cuadro deberá tener 100px de tamaño, por ello a la hora de modificar la matriz se han sumando +100. Además de que el incremento de cada fila es de 200 por esta misma razón, saltando un cuadro de por medio.

Posteriormente, se le paso ese mismo enunciado a ChatGPT-5.6 Luna para la resolución de dicho ejercicio, con la única instrucción adicional de que resolviese el problema de la forma más eficiente posible. Este modelo optó por generar el tablero de manera vectorizada con NumPy, determinando si la casilla es negra o blanca por la paridad de la fila y de la columna.

### Tarea 2, Cuadro de Mondrian

Tanto para un caso como otro, se estuvo "jugando" con las distintas utilidades de OpenCV de forma que quedará lo más afín al estilo Mondrian posible, sin copiar en exceso a la imagen de referencia.


### Tarea 3, Detección de pixel más claro/oscuro

En esta tarea, se ha usado IA debido al desconocimiento de los métodos que pudieran ser útiles en este preciso caso. Aún así, se realizó un approach "bruto" y sencillo de recorrer la imagen en su totalidad para la búsqueda de estos pixeles. Esto, como es lógico, genera una gran cantidad de retraso en el muestreo de frame a frame, tardando varios segundos entre frame y frame.

Por ello, se decidió preguntar a la IA sobre este acercamiento, cómo solucionarlo y que código propondría para este caso. Las diferencias son claras, mientras que el approach anterior implicaba una gran cantidad de operaciones de Python, la IA convirtió la imagen a monocromático y se la pasó a un método de OpenCV. Éste último método "recorre" la imagen de una pasada, y al estar compilado en C, es mucho más rápido y eficiente. El método devuelve los valores de los 2 pixeles más claros y oscuros, además de su posición en el frame.

Con esta información, sólo hacía falta agregar los círculos en dichas posiciones. 


### Tarea 4, Propuesta de Pop Art

Para esta tarea hemos decidido crear un collage 3x3, a diferencia del collage inicial 2x2, donde simplemente se invertían los valores de los canales RGB, y en este nuevo Pop Art se varían los canales de forma que se invierten, oscurecen y se aclaran.

Para ello, se preguntó a la IA previamente sobre diferentes formas de cambiar los canales de salida y cómo se podían variar de forma que causaran diferentes efectos y colores, y a partir de dicha explicación proporcionada por la IA se fue jugando hasta dar con una combinación que nos gustara como propuesta.

## Conversaciones con IA

Para la **tarea 3**, al ser un chat temporal, no se pudo guardar la conversación. Aun así, la conversación se basó meramente en notificar los errores del código inicial (el bucle doble) y sugerir una versión totalmente distinta ya explicada en el razonamiento de la tarea correspondiente.

Conversación para la **tarea 4**: https://claude.ai/share/a37b233b-2c96-4fb5-9c36-8146263758e8

## Referencias

Enlace a una lista de reproducción de videos sobre OpenCV, se han visto un par de videos para entender la dinámica de la librería, [enlace](https://www.youtube.com/playlist?list=PLzMcBGfZo4-lUA8uGjeXhBUUzPYc6vZRn)

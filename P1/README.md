# Práctica 1. Primeros pasos con OpenCV

### Contenidos

**Descripción** 
**Planteamiento de cada tarea**
**Conversaciones con IA**

### Descripción  

En esta práctica se han llevado a cabo una serie de ejercicios básicos para una toma de contacto con la librería OpenCV, así cómo las operaciones y estructura general. Las tareas llevadas a cabo son las siguientes:

- Realizar un tablero de ajedrez con una imagen 800x800 de forma manual. Una vez hecho, comparar con una versión de la IA.
- Realizar una imagen con el estilo del "Cuadro de Mondrian" usando herramientas del OpenCV. Sin usar IA. [Link de referencia](https://www3.gobiernodecanarias.org/medusa/ecoescuela/sa/2017/04/17/descubriendo-a-mondrian/)
- Detectar el pixel más oscuro y más claro de cada fotograma obtenida mediante una cámara de video. En ambos pixeles, añadir un círculo para identificarlos. Si se ha usado IA, mostrar la conversación.
- Realizar una propuesta de Pop Art usando la cámara. En caso de usar IA, incluir la conversación.
  
### Tarea 1, Tablero de ajedrez

Para este ejercicio, se ha planteado realizar un bucle doble, donde el primero representa las columnas y el segundo las filas. Como el color por defecto era el negro, para saber cuando colorear el primer cuadro empezando desde la parte superior, se comprobó si el valor de las centenas módulo 2 es 0. En ese caso, se empieza a colorear el primer cuadro y en caso contrario, se deja en negro y se empieza en el siguiente cuadro debajo.

Al ser una imagen de 800x800, cada cuadro deberá tener 100px de tamaño, por ello a la hora de modificar la matriz se han sumando +100. Además de que el incremento de cada fila es de 200 por esta misma razón, saltando un cuadro de por medio.

Posteriormente, se le paso ese mismo enunciado a ChatGPT-5.6 Luna para la resolución de dicho ejercicio, con la única instrucción adicional de que resolviese el problema de la forma más eficiente posible. Este modelo optó por generar el tablero de manera vectorizada con NumPy, determinando si la casilla es negra o blanca por la paridad de la fila y de la columna.

### Tarea 2, Cuadro de Mondrian

Tanto para un caso como otro, se estuvo "jugando" con las distintas utilidades de OpenCV de forma que quedará lo más afín al estilo Mondrian posible, sin copiar en exceso a la imagen de referencia.


### Tarea 3, Detección de pixel más claro/oscuro

En esta tarea, se ha usado IA debido al desconocimiento de los métodos que pudieran ser útiles en este preciso caso. Aún así, se realizó un approach "bruto" y sencillo de recorrer la imagen en su totalidad para la búsqueda de estos pixeles. Esto, como es lógico, genera una gran cantidad de retraso en el muestreo de frame a frame, tardando varios segundos entre frame y frame.

Por ello, se decidió preguntar a la IA sobre este acercamiento, cómo solucionarlo y que código propondría para este caso. Las diferencias son claras, mientras que el approach anterior implicaba una gran cantidad de operaciones de Python, la IA convirtió la imagen a monocromático y se la pasó a un método de OpenCV. Éste último método "recorre" la imagen de una pasada, y al estar compilado en C, es mucho más rápido y eficiente. El método devuelve los valores de los 2 pixeles más claros y oscuros, además de su posición en el frame.

Con esta información, sólo hacía falta agregar los círculos en dichas posiciones. 


### Tarea 4, Propuesta de Pop Art

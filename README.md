# Reconocimiento_Emociones
Pyhton script with a neuronal network to recognize emotions
# Reconocimiento_Emociones by Jorge Gil
Pyhton script with a neuronal network to recognize emotions

🔍 Resumen del funcionamiento del script
Este script esta diseñado para:

Capturar video en tiempo real con OpenCV desde la cámara.
Detectar rostros en cada fotograma.
Reconocer la emoción de la persona en función de la expresión facial.
Cargar una imagen de emoji que represente la emoción detectada.
Mostrar en pantalla la imagen de la cámara junto con el emoji correspondiente a la emoción.

📸 Captura de video
El código usa cv2.VideoCapture(0) para acceder a la cámara y leer cada fotograma con cap.read().
si es necesario cambia el 0 por el 1 dependiendo de cual camara vayas a usar.

🔍 Detección de rostro
se utiliza un modelo pre-entrenado como Haar cascades o DNN con OpenCV para detectar la cara en la imagen.
Una vez detectado el rostro, se recorta de la imagen y se pone en escala de grises para enviarlo al modelo de reconocimiento de emociones.

😊 Reconocimiento de emociones
El script usa un modelo de deep learning (modelo entrenado con Keras/TensorFlow o red neuronal convolucional).
Este modelo recibe la imagen del rostro y devuelve una emoción como 'feliz', 'triste', 'enojado','asco', 'sorpresa' , 'neutral', segun sea detectado en la camara.

🖼️ Carga y visualización de emojis
Dependiendo de la emoción detectada, se carga una imagen desde una carpeta (Data/Emojis/).

Se usa cv2.imread() para cargar el emoji correspondiente.

Se concatenan las imágenes con cv2.hconcat() para mostrar la imagen de la cámara junto con el emoji.

🖥️ Visualización con OpenCV
Se usa cv2.imshow() para mostrar el resultado.

Un bucle infinito (while True) mantiene el video en ejecución hasta que el usuario presiona una tecla para salir (cv2.waitKey(1)).

_____________________________________________________________________________________________________________________________________________

🔍 Overview of the Script's Functionality
This script is designed to:

Capture real-time video with OpenCV from the camera.
Detect faces in each frame.
Recognize the person's emotion based on their facial expression.
Load an emoji image that represents the detected emotion.
Display the camera image on the screen along with the corresponding emoji for the emotion.

📸 Video Capture
The code uses cv2.VideoCapture(0) to access the camera and read each frame with cap.read().
If needed, change 0 to 1 depending on which camera you will be using.

🔍 Face Detection
A pre-trained model like Haar cascades or DNN with OpenCV is used to detect the face in the image.
Once the face is detected, it is cropped from the image and converted to grayscale to be sent to the emotion recognition model.

😊 Emotion Recognition
The script uses a deep learning model (a model trained with Keras/TensorFlow or a convolutional neural network).
This model receives the face image and returns an emotion such as 'happy', 'sad', 'angry', 'disgust', 'surprise', 'neutral', based on what is detected in the camera.

🖼️ Loading and Displaying Emojis
Depending on the detected emotion, an image is loaded from a folder (Data/Emojis/).

cv2.imread() is used to load the corresponding emoji.

The images are concatenated with cv2.hconcat() to display the camera image alongside the emoji.

🖥️ Display with OpenCV
cv2.imshow() is used to show the result.

An infinite loop (while True) keeps the video running until the user presses a key to exit (cv2.waitKey(1)).

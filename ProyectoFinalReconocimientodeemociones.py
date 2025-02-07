# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:03:27 2022
Proyecto Final de reconocimiento emocional de rostros
@author: jorgil
"""
import time
import cv2
import os
import imutils
import numpy as np
dataPath = 'C:/Users/Python Scripts/Reconocimieno de Emociones/Reconocimiento_Emociones/Data' #Cambia a la ruta donde hayas almacenado Data
def captura_de_emociones(emotionName):
    emotionsPath = os.path.join(dataPath, emotionName)

    if not os.path.exists(emotionsPath):
        print(f'Carpeta creada: {emotionsPath}')
        os.makedirs(emotionsPath)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
            cv2.imwrite(os.path.join(emotionsPath, f'rostro_{count}.jpg'), rostro)
            count = count + 1

        cv2.imshow(emotionName, frame)

        k = cv2.waitKey(1)
        if k == 47 or count >= 50:  # ESC o 30 imágenes capturadas
            break

    cap.release()
    cv2.destroyAllWindows()

def obtenerModelo(method,facesData,labels):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	print("Entrenando ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	emotion_recognizer.write("modelo"+method+".xml")

def entrenamiento():
    emotionsList = os.listdir(dataPath)
    print('Lista de emociones:', emotionsList)
    labels = []
    facesData = []
    label = 0

    for nameDir in emotionsList:
        emotionsPath = os.path.join(dataPath, nameDir)

        for fileName in os.listdir(emotionsPath):
            imgPath = os.path.join(emotionsPath, fileName)
            print('Procesando imagen:', imgPath)
            image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Advertencia: No se pudo cargar la imagen {imgPath}.")
                continue
            image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
            facesData.append(image)
            labels.append(label)

        label = label + 1

    cv2.destroyAllWindows()
    obtenerModelo('EigenFaces', facesData, labels)
    obtenerModelo('FisherFaces', facesData, labels)
    obtenerModelo('LBPH', facesData, labels)

def emotionImage(emotion):
	# Emojis
    if emotion == 'Felicidad': image = cv2.imread('Data/Emojis/felicidad.jpeg') #Verifica donde tienes la carpeta de emojis
    if emotion == 'Enojo': image = cv2.imread('Data/Emojis/enojo.jpeg')
    if emotion == 'Sorpresa': image = cv2.imread('Data/Emojis/sorpresa.jpeg')
    if emotion == 'Tristeza': image = cv2.imread('Data/Emojis/tristeza.jpeg')
    if emotion == 'Asco': image = cv2.imread('Data/Emojis/asco.jpeg')
    if emotion == 'Neutral': image = cv2.imread('Data/Emojis/Neutral.jpeg')
    return image


choice=input('Deseas Tomar las fotos? (Y/N)\n')
if choice=='y' or choice=='Y':
    print("Tomaremos las imagenes para entrenar el modelo:")
    lista=['Enojo','Felicidad','Sorpresa','Tristeza','Neutral','Asco']
    time.sleep(2)
    for e in lista:
        time.sleep(2)
        print('Pon cara de: ',e)
        emotionName = e
        captura_de_emociones(emotionName)

choice1=input('Deseas entrenar el modelo? (Y/N)\n')
if choice1=='y' or choice1=='Y':
    print('entrenando')
    entrenamiento()


mod=input('Elige un metodo de entranamiento y lectura del modelo\n 1.-EigenFaces\n 2.-FisherFaces\n 3.-LBPH\n\n')
if mod == '1': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if mod == '2': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if mod == '3': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

if mod=='1' : 
    method = 'EigenFaces'
    emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if mod=='2':
    method = 'FisherFaces'
    emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if mod=='3':
    method = 'LBPH'
    emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')

imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:

	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = emotion_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

		# EigenFaces
		if method == 'EigenFaces':
			if result[1] < 5700:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				if image is not None:
					image = cv2.resize(image, (300, frame.shape[0]))
					if len(image.shape) == 2:  
						image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
					nFrame = cv2.hconcat([frame, image])
				else:
					nFrame = cv2.hconcat([frame, np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
		
		# FisherFace
		if method == 'FisherFaces':
			if result[1] < 500:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
                if image is not None:
					image = cv2.resize(image, (300, frame.shape[0]))
					if len(image.shape) == 2:  
						image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
					nFrame = cv2.hconcat([frame, image])
				else:
					nFrame = cv2.hconcat([frame, np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
		
		# LBPHFace
		if method == 'LBPH':
			if result[1] < 60:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
                if image is not None:
					image = cv2.resize(image, (300, frame.shape[0]))
					if len(image.shape) == 2:  
						image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
					nFrame = cv2.hconcat([frame, image])
				else:
					nFrame = cv2.hconcat([frame, np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])

	cv2.imshow('nFrame',nFrame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()

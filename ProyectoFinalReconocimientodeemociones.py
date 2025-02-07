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
dataPath = 'C:/Users/jorgil/Documents/Python Scripts/Reconocimieno de Emociones/Data' #Cambia a la ruta donde hayas almacenado Data
def captura_de_emociones(emotionName):
    #dataPath = 'C:/Users/jorgil/Documents/Python Scripts/Reconocimieno de Emociones/Data' #Cambia a la ruta donde hayas almacenado Data
    emotionsPath = dataPath + '/' + emotionName

    if not os.path.exists(emotionsPath):
    	print('Carpeta creada: ',emotionsPath)
    	os.makedirs(emotionsPath)

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0

    while True:

    	ret, frame = cap.read()
    	if ret == False: break
    	frame =  imutils.resize(frame, width=640)
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	auxFrame = frame.copy()

    	faces = faceClassif.detectMultiScale(gray,1.3,5)

    	for (x,y,w,h) in faces:
    		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    		rostro = auxFrame[y:y+h,x:x+w]
    		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
    		cv2.imwrite(emotionsPath + '/rotro_{}.jpg'.format(count),rostro)
    		count = count + 1
    	cv2.imshow(emotionName,frame)

    	k =  cv2.waitKey(1)
    	if k == 27 or count >= 30:
    		break

    cap.release()
    cv2.destroyAllWindows()

def obtenerModelo(method,facesData,labels):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo"+method+".xml")

def entrenamiento():
    #dataPath = 'C:/Users/jorgil/Documents/Python Scripts/Reconocimieno de Emociones/Data' #Cambia a la ruta donde hayas almacenado Data
    emotionsList = os.listdir(dataPath)
    print('Lista de emociones: ', emotionsList)
    labels = []
    facesData = []
    label = 0
    for nameDir in emotionsList:
    	emotionsPath = dataPath + '/' + nameDir
    	for fileName in os.listdir(emotionsPath):
    		print('Rostros: ', nameDir + '/' + fileName)
    		labels.append(label)
    		facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))
    		image = cv2.imread(emotionsPath+'/'+fileName,0)
    #		cv2.imshow('image',image)
    	label = label + 1
    cv2.destroyAllWindows()
    obtenerModelo('EigenFaces',facesData,labels)
    obtenerModelo('FisherFaces',facesData,labels)
    obtenerModelo('LBPH',facesData,labels)

def emotionImage(emotion):
	# Emojis
    if emotion == 'Felicidad': image = cv2.imread('Emojis/felicidad.jpeg')
    if emotion == 'Enojo': image = cv2.imread('Emojis/enojo.jpeg')
    if emotion == 'Sorpresa': image = cv2.imread('Emojis/sorpresa.jpeg')
    if emotion == 'Tristeza': image = cv2.imread('Emojis/tristeza.jpeg')
    if emotion == 'Asco': image = cv2.imread('Emojis/asco.jpeg')
    if emotion == 'Neutral': image = cv2.imread('Emojis/Neutral.jpeg')
    return image


choice=input('Deseas Tomar las fotos? (Y/N)\n')
if choice=='y' or choice=='Y':
    print("Tomaremos las imagenes para entrenar el modelo:")
    lista=['Enojo','Felicidad','Sorpresa','Tristeza','Neutral','Asco']
    #print(lista)
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
# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
#method = 'EigenFaces'
#method = 'FisherFaces'
#method = 'LBPH'

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
# --------------------------------------------------------------------------------

#dataPath = 'C:/Users/jorgil/Documents/Python Scripts/Reconocimieno de Emociones/Data' #Cambia a la ruta donde hayas almacenado Data
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

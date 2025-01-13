# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:09:55 2024

@author: Jasmine Moreira
"""

import cv2
import time
import numpy as np
import mediapipe as mp

# Inicialização do MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Captura de vídeo da câmera
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

# Lista para armazenar os dados das posições dos dedos
x_finger_positions = []
y_finger_positions = []
z_finger_positions = []

rval = True
go = False
while rval:
    if not go:
        rval, frame = vc.read()  
        cv2.imshow("preview", frame)
            
    key = cv2.waitKey(20)
    if key == 27:
        break
    if key == ord('c'): # or go == True: # capturar e sair
        #go = True
        # Captura de 30 imagens
        for _ in range(30):
            rval, frame = vc.read()
            cv2.imshow("preview", frame)
            key = cv2.waitKey(20)
            if not rval:
                print("Erro ao capturar imagem da câmera.")
                break
            # Processamento da imagem com o MediaPipe Hand Landmarker
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Coleta das posições dos dedos (pontos landmarks)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                x_finger_positions.append([landmark.x for landmark in landmarks])
                y_finger_positions.append([landmark.y for landmark in landmarks])
                z_finger_positions.append([landmark.z for landmark in landmarks])
                
        # Conversão da lista para uma matriz numpy
        finger_matrix = np.array(x_finger_positions+y_finger_positions+z_finger_positions)
        # Salva a matriz em um arquivo numerado com base no timestamp
        timestamp = int(time.time())
        filename = f"data/A_white/matrix_{timestamp}.png"
        #plt.imsave(filename, finger_matrix, cmap='gray')
        print(f"Capturadas {finger_matrix.shape[0]} amostras e salvas em {filename}.")
        x_finger_positions = []
        y_finger_positions = []
        z_finger_positions = []
        
    
cv2.destroyWindow("preview")
vc.release()



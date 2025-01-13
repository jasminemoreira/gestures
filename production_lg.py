# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:23:22 2024

@author: Jasmine Moreira
"""

import cv2
import tinytuya
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

speed = 3 # integer 1,2,3 
# setup de lampada
d = tinytuya.BulbDevice('20584013483fda833e7f', '192.168.18.21', "I}'Gah*eeisLu0{R")
d.set_version(3.3)
d.set_colour(0, 0, 0, True)

# Carregar modelo
base_dir = r"C:\Users\jasmi\OneDrive\Documents\Doutorado II\Disciplinas\Deep Learning\projeto_final"
model = load_model(base_dir+'/model_gestures.h5')

gestures = {0: "ar", 
            1: "cortinas", 
            2: "janelas", 
            3:"luzes", 
            4:"",
            5:"chavear",
            6:"aumentar",
            7:"reduzir",
            8:"vermelho",
            9:"azul",
            10:"branco"}

devices = ("ar", "cortinas", "janelas", "luzes")
state = {"ar": False, "cortinas": False, "janelas": False, "luzes": False}
level = {"ar": 2, "cortinas": 2, "janelas": 2, "luzes": 2}
context = "ar"
color = ""
text_color = (165, 165, 165)

# fundo
overlay_image = cv2.imread('black.jpeg')  
overlay_image = cv2.resize(overlay_image, (1280, 100))
h, w = overlay_image.shape[:2]

# Inicialização do MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Captura de vídeo da câmera
cv2.namedWindow("Gestures")
cv2.resizeWindow('Gestures', 1280, 720)
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Lista para armazenar os dados das posições dos dedos
x_finger_positions = []
y_finger_positions = []
z_finger_positions = []
rval = True
text = ""
while rval:
    rval, frame = vc.read()  
         
    # Processamento da imagem com o MediaPipe Hand Landmarker
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Coleta das posições dos dedos (pontos landmarks)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        for _ in range(0,speed):
            x_finger_positions.append([landmark.x for landmark in landmarks])
            y_finger_positions.append([landmark.y for landmark in landmarks])
            z_finger_positions.append([landmark.z for landmark in landmarks])

    elif len(x_finger_positions):
        x_finger_positions = []  #limpar buffer se a mão sair de cena
        y_finger_positions = []
        z_finger_positions = []
    
    if len(x_finger_positions) == 30:
        # Conversão da lista para uma matriz numpy
        finger_matrix = np.array(x_finger_positions+y_finger_positions+z_finger_positions)
        normalized_array = (finger_matrix - np.min(finger_matrix)) / (np.max(finger_matrix) - np.min(finger_matrix)) * 255
        grayimg = normalized_array.astype(np.uint8)
        grayimg = np.expand_dims(grayimg, axis = 0)
        res = model.predict(grayimg, verbose = 0)
        probcat = max(list(res[0]))
        cat = list(res[0]).index(probcat)
        for _ in range(0,speed):
            x_finger_positions.pop(0)
            y_finger_positions.pop(0)
            z_finger_positions.pop(0)

        if probcat > 0.98:      # regular a sensibilidade
            text = gestures[cat]
                       
    if text == "chavear":
        state[context] = False if state[context] else True
        text_color = (255, 255, 255) if state[context] else (165, 165, 165)
    elif text == "aumentar":
        if state[context]:
            level[context] = 7 if level[context]>=7 else level[context]+1       
    elif text == "reduzir":
        if state[context]:
            level[context] = 1 if level[context]<=1 else level[context]-1
    elif text == "":
        context = context
    elif text in devices:
        context = text 
        text_color = (255, 255, 255) if state[context] else (165, 165, 165)
        
 
    # executar ações
    if context == "luzes" and text !="":    
        if state[context]:          
            if text=="vermelho" or (text in ("luzes", "aumentar", "reduzir") and color=="vermelho"):
                color = "vermelho"
                d.set_colour(int(255*level[context]/7), 0, 0, True)
                text_color = (100, 100, 255)
            elif text=="azul" or (text in ("luzes", "aumentar", "reduzir") and color=="azul"):
                color = "azul"
                d.set_colour(0, 0, int(255*level[context]/7), True)
                text_color = (255, 100, 100)
            elif text=="branco" or color=="" or (text in ("luzes", "aumentar", "reduzir") and color=="branco"):
                color = "branco"
                d.set_white(int(10+165*(level[context]-1)), 255, True)
                text_color = (255, 255, 255)                
        else:
            color = ""
            d.set_colour(0, 0, 0, True)
            text_color = (165, 165, 165)
                   
    if text != "" :
        x_finger_positions = []  
        y_finger_positions = []
        z_finger_positions = []
        
    text = "" # uma vez processado, liberar  
    
    #frame = cv2.flip(frame, 1);
    shapes = np.zeros_like(frame, np.uint8)
    shapes[frame.shape[0]-h:, frame.shape[1]-w:] = overlay_image
    mask = shapes.astype(bool)
    alpha = 0.5
    frame[mask] = cv2.addWeighted(frame, 1 - alpha, shapes, alpha, 0)[mask]
    
    if context != "":
        cv2.putText(img=frame, text=context+": "+">"*level[context], 
                    org=(40, 690), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=3, 
                    color=text_color,thickness=6)
        
    cv2.imshow("Gestures", frame)    
    key = cv2.waitKey(1)
    if key == 27:
        break
           
cv2.destroyWindow("Gestures")
vc.release()
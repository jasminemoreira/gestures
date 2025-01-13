# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:23:22 2024

@author: Jasmine Moreira
"""

import cv2
import pygame
import tinytuya
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import requests
import time


speed = 3 # integer 1,2,3 
slide_step = 6

def send_request(api_url, json_data):
    try:
        response = requests.post(api_url, json=json_data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erro na requisição (código {response.status_code}): {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None

send_request("http://172.21.107.165:8000/BBX_STU_Initialize", {})
send_request("http://172.21.107.165:8000/BBX_STU_OpenTextToSpeech", 
             {"language":"1","pitch":"50","tempo":"50"})

lastT = 0
lastS = ""
def say(text):
    global lastT, lastS
    #return
    deltaT = time.time() - lastT
    lastT = time.time();
    if deltaT > 0.2 and text != lastS:
        send_request("http://172.21.107.165:8000/BBX_STU_PlayTextToSpeech", 
                 {"text":text,"gender":"2","mix":"0"})   
    lastS = text

say("Olá Jasmine, seja bem-vinda ao sistema de reconhecimento de gestos dinâmicos R$ 123.456.789,12")


pygame.init()
music_path = 'musica.mp3'
pygame.mixer.music.load(music_path)
pygame.mixer.music.play(-1)
volume = 0.0
pygame.mixer.music.set_volume(volume)

# setup de lampada
d = tinytuya.BulbDevice('20584013483fda833e7f', '192.168.18.21', "I}'Gah*eeisLu0{R")
d.set_version(3.3)
d.set_colour(0, 0, 0, True)

# Carregar modelo
base_dir = r"C:\Users\jasmi\OneDrive\Documents\Doutorado II\Disciplinas\Deep Learning\projeto_final"
model = load_model(base_dir+'/model_gestures.h5')

gestures = {0: "som", 
            1: "cortina", 
            2: "janela", 
            3:"luz", 
            4:"",
            5:"chavear",
            6:"aumentar",
            7:"reduzir",
            8:"vermelho",
            9:"azul",
            10:"branco"}

devices = ("som", "cortina", "janela", "luz")
state = {"som": False, "cortina": False, "janela": False, "luz": False}
level = {"som": 1, "cortina": 2, "janela": 2, "luz": 2}
context = "som"
color = ""
text_color = (165, 165, 165)

# fundo
overlay_image = cv2.imread('black.jpeg')  
overlay_image = cv2.resize(overlay_image, (640, 80))
h, w = overlay_image.shape[:2]

# Inicialização do MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Captura de vídeo da câmera
cv2.namedWindow("Gestures")
cv2.resizeWindow('Gestures', 1400, 1050)
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_AUTOFOCUS, 0) 

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
        for _ in range(0, slide_step):
            x_finger_positions.pop(0)
            y_finger_positions.pop(0)
            z_finger_positions.pop(0)

        if probcat > 0.99:      # regular a sensibilidade
            text = gestures[cat]
                       
    if text == "chavear":
        state[context] = False if state[context] else True
        text_color = (255, 255, 255) if state[context] else (165, 165, 165)
        if state[context]:
            say("dispositivo ligado")
        else:
            say("dispositivo desligado")
    elif text == "aumentar":
        if state[context]:
            level[context] = 7 if level[context]>=7 else level[context]+1
            say("intensidade aumentada")
    elif text == "reduzir":
        if state[context]:
            level[context] = 1 if level[context]<=1 else level[context]-1
            say("intensidade reduzida")
    elif text == "":
        context = context
    elif text in devices:
        say(text) if text != "som" else say("som ambiente")
        context = text 
        text_color = (255, 255, 255) if state[context] else (165, 165, 165)
        
 
    # executar ações
    if context == "luz" and text !="":    
        if state[context]:          
            if text=="vermelho" or (text in ("luzes", "aumentar", "reduzir") and color=="vermelho"):
                color = "vermelho"
                d.set_colour(int(255*level[context]/7), 0, 0, True)
                text_color = (100, 100, 255)
                if text == "vermelho":
                    say("cor alterada para vermelho")
            elif text=="azul" or (text in ("luzes", "aumentar", "reduzir") and color=="azul"):
                color = "azul"
                d.set_colour(0, 0, int(255*level[context]/7), True)
                text_color = (255, 100, 100)
                if text == "azul":
                    say("cor alterada para azul")
            elif text=="branco" or color=="" or (text in ("luzes", "aumentar", "reduzir") and color=="branco"):
                color = "branco"
                d.set_white(int(10+165*(level[context]-1)), 255, True)
                text_color = (255, 255, 255)
                if text == "branco":
                    say("cor alterada para branco")
        else:
            color = ""
            d.set_colour(0, 0, 0, True)
            text_color = (165, 165, 165)
            
    if context== "som" and text != "":
        if state[context]:        
            pygame.mixer.music.set_volume(level[context]/7)
        else:
            pygame.mixer.music.set_volume(volume)
                   
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
                    org=(40, 450), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=2, 
                    color=text_color,thickness=3)
    frame = cv2.resize(frame, (1400, 1050)) 
    cv2.imshow("Gestures", frame)    
    key = cv2.waitKey(1)
    if key == 27:
        break
           
cv2.destroyWindow("Gestures")
vc.release()
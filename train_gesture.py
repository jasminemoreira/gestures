# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:32:28 2024

@author: Jasmine Moreira

1) Preparar dados
2) Criar o modelo (input, output size, forward pass)
3) Criar a função de erro (loss) e o otimizador 
4)  Executar treinamento
"""
import os
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preparação dos dados
base_dir = r"C:\Users\jasmi\OneDrive\Documents\Doutorado II\Disciplinas\Deep Learning\projeto_final"

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir,"data"),
        target_size=(90,21),
        batch_size=8,
        class_mode="categorical"
        )

validation_generator = validation_datagen.flow_from_directory(
        os.path.join(base_dir,"data_validation"),
        target_size=(90,21),
        batch_size=8,
        class_mode="categorical"
        )

# criar o modelo
model = models.Sequential()
model.add(layers.Conv2D(33, (3,3),activation='relu', input_shape=(90,21,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l1(0.0001)))
model.add(layers.Dense(11, activation = 'softmax'))

model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

#import visualkeras
#visualkeras.layered_view(model).show()

history = model.fit(train_generator, 
                    epochs=150, 
                    validation_data=validation_generator)

model.save(base_dir+'\model_gestures.h5')

# avaliar resultado do treinamento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
























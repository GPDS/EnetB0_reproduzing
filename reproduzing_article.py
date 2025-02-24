import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D, Activation, MaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import confusion_matrix, classification_report
import os

# Diretórios do dataset (Substitua pelos seus caminhos)
train_dir = "Train"
val_dir = "Validation"
test_dir = "Test"

# Parâmetros do modelo
IMG_SIZE = (224, 224)  # Tamanho da imagem compatível com EfficientNetB0
BATCH_SIZE = 40

# Data Augmentation para aumentar a diversidade dos dados
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalização
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #horizontal_flip=True,
    #zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Geradores de imagens
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


#Carregar EfficientNetB0 sem a camada de classificação
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do EfficientNetB0 para evitar treinar desde o início
base_model.trainable = False

# Construção do modelo conforme solicitado
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Redução da dimensionalidade
x = BatchNormalization()(x)  # ✅ Batch Normalization após EfficientNetB0
x = Dense(256, activation='softmax')(x)  # ✅ Camada totalmente conectada
x = Dropout(0.1)(x)  # ✅ Regularização para evitar overfitting
output_layer = Dense(2, activation='softmax')(x)  # ✅ Camada de saída

# input_shape = (224, 224, 3)
# inputs = Input(shape=input_shape)
    
# x = Conv2D(32, (3, 3), padding='same')(inputs)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)

# x = Conv2D(64, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)

# x = Conv2D(128, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)

# x = Flatten()(x)
# x = Dense(512)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dropout(0.5)(x)
# outputs = Dense(2, activation='softmax')(x)


# Criar o modelo final
model = Model(base_model.input, output_layer)

# Compilar o modelo
model.compile(optimizer=Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])




# Exibir o resumo do modelo
model.summary()

EPOCHS = 40

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)


# Avaliação nos dados de teste
test_loss, test_acc = model.evaluate(test_generator)
print(f"Acurácia no Teste: {test_acc * 100:.2f}%")

# Previsões no conjunto de teste
loss, acc = model.evaluate(test_generator)
#y_pred = np.round(y_pred).flatten()

print(f"A ACURACIA DO MODELO É: {acc}")


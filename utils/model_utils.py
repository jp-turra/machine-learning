import time
import os

import matplotlib.pyplot as plt

import numpy as np

import cv2

from keras.api.models import Model, model_from_json, Sequential
from keras.api.layers import Resizing, Rescaling, RandomFlip, RandomRotation
from keras.api.callbacks import History

from cv2.typing import MatLike

def salvar_modelo(modelo: Model, path: os.PathLike):
    nome_modelo = modelo.name
    if not nome_modelo:
        nome_modelo = 'model_' + time.strftime("%Y%m%d_%H%M%S")
        modelo.name = nome_modelo
    
    modelo_json = modelo.to_json()
    with open(os.path.join(path, f"{nome_modelo}.model.json"), 'wt') as json:
        json.write(modelo_json)
    
    modelo.save_weights(
        os.path.join(path, f"{nome_modelo}.weights.h5")
    )

def carregar_modelo(dir_path: os.PathLike, nome_modelo: str):
    with open(os.path.join(dir_path, f"{nome_modelo}.model.json"), 'r') as json:
        modelo: Model = model_from_json(json.read())
    
    modelo.load_weights(os.path.join(dir_path, f"{nome_modelo}.weights.h5"))

    return modelo

def print_history(history: History):
    if not history or not history.history:
        return
    
    if history.history.get('accuracy'):
        print("Precisão de treinamento: ", np.mean(history.history['accuracy']) * 100, "%")

    if history.history.get('val_accuracy'):
        print("Precisão de teste: ", np.mean(history.history['val_accuracy']) * 100, "%")

    if history.history.get('loss'):
        print("Perda de treinamento: ", np.mean(history.history['loss']))

    if history.history.get('val_loss'):
        print("Perda de teste: ", np.mean(history.history['val_loss']))

def plot_history(history: History):
    if not history or not history.history:
        return
    
    display = False
    _, (ax1, ax2) = plt.subplots(
        figsize=(9, 5),
        nrows=1,
        ncols=2
    )
    ax1.set_title("Acurácia")
    ax2.set_title("Custo")

    if history.history.get('accuracy'):
        display = True
        ax1.plot(history.history['accuracy'], 'ro-', label="Acc. Treinamento")

    if history.history.get('val_accuracy'):
        display = True
        ax1.plot(history.history['val_accuracy'], 'go-', label="Acc. Teste")

    if history.history.get('loss'):
        display = True
        ax2.plot(history.history['loss'], 'ro-', label="Loss Treinamento")

    if history.history.get('val_loss'):
        display = True
        ax2.plot(history.history['val_loss'], 'go-', label="Loss Teste")

    if display:
        ax1.legend()
        ax2.legend()
        plt.show()

def plot_image(image: np.ndarray, title: str = "", cmap: str | None = 'gray',
                figsize: tuple[int, int] = (10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(f'Classe: {title}')

    plt.show()

def plot_random_images(image_dataset: np.ndarray, class_dataset: np.ndarray | None = None, 
                        cmap: str | None = 'gray', figsize: tuple[int, int] = (10, 10)):
    plt.figure(figsize=figsize)

    index = np.random.randint(0, image_dataset.shape[0])
    image = image_dataset[index]
    if class_dataset:
        class_name = class_dataset[index]
        plt.title(f'Classe: {class_name}')

    plt.imshow(image, cmap=cmap)
    plt.show()

def display_image(window_name: str, image: MatLike):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def change_image_color(image: MatLike, color: int = cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(image, color)

def resize_rescale_image(images, height: int, width: int, dtype: np.dtype = np.float32, scale_factor: float = 1./255):
    resize_rescale = Sequential([
        Resizing(height=height, width=width, dtype=dtype),
        Rescaling(scale_factor)
    ])

    return resize_rescale(images)

def augment_data(image, number_of_augmentations: int, flip_direction: str = 'horizontal_and_vertical', rotation_factor: float = 0.2):
    data_augmentation = Sequential([
        RandomFlip(flip_direction),
        RandomRotation(rotation_factor),
    ])
    images = []
    for i in range(number_of_augmentations):
        augmented_image = data_augmentation(image)
        images.append(augmented_image)

    return images

import time
import os

import matplotlib.pyplot as plt

import numpy as np

import cv2

from keras.api.callbacks import History
from keras.api.layers import Resizing, Rescaling, Reshape
from keras.api.layers import RandomFlip, RandomRotation, RandomBrightness, RandomContrast, RandomCrop, RandomTranslation, RandomZoom
from keras.api.models import Model, model_from_json, Sequential
from keras.api.preprocessing import image_dataset_from_directory

from tensorflow._api.v2.image import convert_image_dtype
from tensorflow._api.v2.data.experimental import AUTOTUNE

from itertools import product

from cv2.typing import MatLike

from numba import cuda

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

def print_history(history: History, accuracy_name='accuracy', loss_name='loss'):
    if not history or not history.history:
        return
    
    if history.history.get(f'{accuracy_name}'):
        print("Precisão de treinamento: ", np.mean(history.history[f'{accuracy_name}']) * 100, "%")

    if history.history.get(f'val_{accuracy_name}'):
        print("Precisão de teste: ", np.mean(history.history[f'val_{accuracy_name}']) * 100, "%")

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

def plot_learning_curve(acc: list, val_acc: list, loss: list, val_loss: list, epochs: list = None, index_range: tuple[int, int] = (0, -1)):
    if not epochs:
        epochs = np.arange(0, len(acc), 1)

    fig, (ax1, ax2) = plt.subplots(
        figsize=(9, 5),
        nrows=1,
        ncols=2
    )

    first_index = index_range[0]
    last_index = index_range[1]

    ax1.plot(epochs[first_index:last_index], acc[first_index:last_index], 'ro-', label="Train Accuracy")
    ax1.plot(epochs[first_index:last_index], val_acc[first_index:last_index], 'go-', label="Test Accuracy")
    ax1.legend()

    ax2.plot(epochs[first_index:last_index], loss[first_index:last_index], 'ro-', label="Train Loss")
    ax2.plot(epochs[first_index:last_index], val_loss[first_index:last_index], 'go-', label="Test Loss")
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

def convert_image_to_float(image, label, dtype='float32'):
    image = convert_image_dtype(image, dtype=dtype)
    return image, label

def load_tf_images_from_directory(dir_path: os.PathLike, image_shape=(224, 224), labels: list | str = "inferred", 
                               label_mode: str = 'binary', shuffle: bool = True, batch_size: int = 32, 
                               interpolation: str = 'bilinear'):
    ds = image_dataset_from_directory(
        directory=dir_path,
        labels=labels,
        label_mode=label_mode,
        image_size=image_shape,
        batch_size=batch_size,
        shuffle=shuffle,
        interpolation=interpolation
    )

    ds = (ds.map(convert_image_to_float).cache().prefetch(buffer_size=AUTOTUNE))

    return ds

def load_image_from_directory(dir_path: os.PathLike, class_names: list, image_shape=(224, 224)):
    if not os.path.exists(dir_path):
        return [], []
    
    imagens = []
    labels = []

    for item in os.listdir(dir_path):
        if item in class_names and os.path.isdir(os.path.join(dir_path, item)):
            for file in os.listdir(os.path.join(dir_path, item)):
                image_path = os.path.join(dir_path, item, file)
                if os.path.isfile(image_path):
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:         
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            if image_shape:
                                image = cv2.resize(image, image_shape)

                            imagens.append(image)
                            labels.append(item)
                    except:
                        pass
    try:
        return np.array(imagens), np.array(labels)
    except:
        return imagens, np.array(labels)
    
def show_kernel(kernel, label=True, digits=None, text_size=28):
    # Format kernel
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)

    # Plot kernel
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # Optionally, add value labels
    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            plt.text(j, i, val, 
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])

def clear_gpu_memory():
    device = cuda.get_current_device()
    if device:
        device.reset()

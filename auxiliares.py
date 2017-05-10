import os
import numpy as np
from scipy.misc import imread

def read_csv(csv_file):
    ''' Abre el archivo csv_file y lo convierte en numpy array'''
    
    array = []
    with open(csv_file) as data_file:
        for line in data_file:
            array.append(map(float,line.strip().split(',')))
        #if len(array) == 1:
        #    array = array[0]
    return np.asarray(array)

def load_dir_images(path):
    ''' Carga las imagenes del directorio path y las devuelve en la lista images'''
    images = [imread(os.path.join(path,f)).dot([0.299, 0.587, 0.114]) for f in os.listdir(path)]
    return images
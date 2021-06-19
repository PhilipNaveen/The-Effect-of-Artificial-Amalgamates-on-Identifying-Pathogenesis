 from skimage import measure
import matplotlib.pyplot as plt
import numpy as nm
import cv2
import random
import os
import tensorflow as tf

def mse(imageA, imageB):
    err = nm.sum((imageA.astype("float") - imageB.astype("float")) ** 2); err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_dataset(data):
    vals = []
    for point in data:
        vals.append(mse(point, data[random.randint(0, len(data)-1)]))
    return vals

def gaussian_distribution(x, dataset):
    mean, sd = float(nm.mean(dataset)), float(nm.std(dataset))
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density
    
def load_images_and_labels(path, classification=None, x=[], y=[]):
    x, y = list(x), list(y)
    for subdir, dirs, files in os.walk(path):
        for file in files: 
            x.append(nm.array(resize(cv2.imread(subdir + "\\" + file))))
            y.append([classification])
    return nm.array(x), nm.array(y)

def load_images(path, x=[]):
    x = list(x)
    for subdir, dirs, files in os.walk(path):
        for file in files: 
            x.append(nm.array(resize(cv2.imread(subdir + "\\" + file))))
    return nm.array(x)
            
def resize(image, IMG_SIZE=100):
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE))

def extract(model, images):
    points = []
    for image in images: points.append(model.predict(image.reshape(1, 100, 100, 3))[0][0])
    return points

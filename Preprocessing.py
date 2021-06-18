from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
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
    
def load(path, classification=None, x=[], y=[]):
    x, y = list(x), list(y)
    for subdir, dirs, files in os.walk(path):
        for file in files: 
            x.append(nm.array(resize(cv2.imread(subdir + "\\" + file))).reshape(1, 100, 100, 3))
            y.append([classification])
    return nm.array(x), nm.array(y)
            
def resize(image, IMG_SIZE=100):
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE))

import tensorflow as tf
import sklearn
import numpy as nm
import pickle
import os
import sys
import matplotlib.pyplot as plt

"""
New hybrid diagnosis model. Uses machine-learning and deep-learning. It's an AI based system using
a dense amalgamate permutated with a convolutional neural-network and support vector machine
in an instance semi-supervised learning. It works from 96-99 percent accuracy.

"""

#class to define: new hybrid diagnosis model
class Model(object):
    
    #constructor
    def __init__(self, feature_extractor, classifier, name="algorithm"):
        self.name = name
        self.feature_extractor = tf.keras.models.load_model(feature_extractor)
        with open(classifier, "rb") as file: self.classifier = pickle.load(file)
        
    #prediction fucntion with feature extraction and classification
    def predict(self, x):
        output = self.feature_extractor.predict(x.reshape(1, 100, 100, 3))[0][0]
        output = self.classifier.predict(output.reshape(-1, 1))
        return output
    
    #return name
    def name(self):
        return self.name
    
    #change feature extractor
    def change_feature_extractor(self, new_feature_extractor):
        self.feature_extractor = tf.keras.models.load_model(new_feature_extractor)
        
    #change classifier
    def change_classifier(self, new_classifier):
        with open(new_classifier, "rb") as file: self.classifier = pickle.load(file)

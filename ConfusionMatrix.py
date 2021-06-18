#import statements
import math
import cmath
import sympy

#class: confusion matrix
class ConfusionMatrix:
  #constructor
  def __init__(self, prediction_series, actual_series):
    self.prediction_series, self.actual_series = prediction_series, actual_series
  
  #method: parse
  def parse(self):
    try:
      self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
      for j in range(len(self.prediction_series)):
        if (self.prediction_series[j] == True and self.actual_series[j] == True):
          self.TP += 1
        elif (self.prediction_series[j] == False and self.actual_series[j] == False):
          self.TN += 1
        elif (self.prediction_series[j] == True and self.actual_series[j] == False):
          self.FP += 1
        elif (self.prediction_series[j] == False and self.actual_series[j] == True):
          self.TN += 1
        else: pass
    except Exception as ERR:
      return ERR
    except KeyboardInterrupt:
      return KeyboardInterrupt

  #method: true positives
  def true_positives(self):
    return self.TP

  #method: true positives
  def true_negatives(self):
    return self.TN

  #method: true positives
  def false_positives(self):
    return self.FP

  #method: true positives
  def true_negatives(self):
    return self.FN

  #method: accuracy
  def accuracy(self):
    return 100 * float((self.TP + self.TN)/(self.TP + self.TN + self.FP + self.TN))
  
  #method: sensitivity
  def sensitivity(self):
    return 100 * float(self.TP/(self.TP + self.FN))

  #method: specificity
  def specificity(self):
    return 100 * float(self.TN/(self.TN + self.FP))

  #method: negative predictive value
  def negative_predictive_value(self):
    return 100 * float(self.TN/(self.TN + self.FN))

  #method: precision
  def precision(self):
    return 100 * float(self.TP/(self.TP + self.FP))

  #method: correct classification rate (CCR)
  def CCR(self): 
    return 100 * float((self.TP + self.TN)/(self.FP + self.FN + self.TP + self.TN))

  #method: to_string
  def to_string(self):
    print("general analysis of data")
    print("true positives: {}".format(self.TP))
    print("true negatives: {}".format(self.TN))
    print("false positives: {}".format(self.FP))
    print("false negatives: {}".format(self.FN))

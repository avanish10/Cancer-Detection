import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense
import tensorflow as tf
from keras.models import Sequential


path = r'C:\Users\AVANISH SINGHAL\Desktop\MinorProject\breast-histopathology-images\8863\0\8863_idx5_x51_y1251_class0.png'

dataset_benign = cv2.imread(path)
cv2.waitKey(1000)
cv2.destroyAllWindows()
cv2.imshow('image',dataset_benign)

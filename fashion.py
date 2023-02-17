import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
import keras
tf.keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()x_train.shape,y_train.shape,'***********************',x_test.shape,y_test.shape
x_train[0]
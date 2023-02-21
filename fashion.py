import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
import keras
tf.keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()x_train.shape,y_train.shape,'***********************',x_test.shape,y_test.shape
x_train[0]
class_labels=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
class_labels
plt.figure(figsize=(16,16))
j=1
for i in np.random.randint(0,1000,25):
  plt.subplot(5,5,j);j+=1
  plt.imshow(x_train[i],cmap="Greys")
  plt.axis('off')
  plt.title('{} / {}'.format(class_labels[y_train[i]],y_train[i]))
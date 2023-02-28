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
  x_train=np.expand_dims(x_train,1)
  x_train.ndim
  x_test=np.expand_dims(x_test,1)
  x_test.ndim
  x_train=x_train/255
x_test=x_test/255
from sklearn.model_selection import train_test_split
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=0.2,random_state=2020)
x_train.shape,x_validation.shape,y_train.shape,y_validation.shape
x_train=np.reshape(x_train,(48000,28,28,1))
x_validation=np.reshape(x_validation,(12000,28,28,1))
x_test.shape
x_test=np.reshape(x_test,(10000,28,28,1))
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
model=keras.models.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=3,strides=(1,1),padding='valid',activation='relu',input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128,activation='relu'),
    keras.layers.Dense(units=10,activation='softmax')
])
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=512,verbose=1,validation_data=(x_validation,y_validation)) 
y_pred=model.predict(x_test)
y_pred.round(2)
y_test
model.evaluate(x_test,y_test)
plt.figure(figsize=(16,16))
j=1
for i in np.random.randint(0,1000,25):
  plt.subplot(5,5,j);j+=1
  plt.imshow(x_test[i].reshape(28,28),cmap='Greys')
  plt.title('Actual= {} / {} \n Predicted {} / {}'.format(class_labels[y_test[i]],y_test[i],class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i])))
  plt.axis('off')
  from sklearn.metrics import confusion_matrix
plt.figure(figsize=(16,9))
y_pred_labels=[np.argmax(label) for label in y_pred]
cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm,annot=True,fmt='d',xticklabels=class_labels,yticklabels=class_labels)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred_labels, target_names=class_labels)
print(cr)
model.save('fashion_mnist_cnn_model.h5')
cnn_model2=keras.models.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=3,strides=(1,1),padding='valid',activation='relu',input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=64,kernel_size=3,strides=(1,1),padding='same',activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128,activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=256,activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=128,activation='relu'),
    keras.layers.Dense(units=10,activation='softmax')
])
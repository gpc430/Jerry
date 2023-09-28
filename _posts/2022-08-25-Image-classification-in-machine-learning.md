## Image classification in machine learning

### Loading training and test data from mnist dataset:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
fashion_minst = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_minst.load_data()
```

### Prepare the Fashion MNIST dataset
```
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_train, X_valid, X_test, y_train, y_valid, y_test 
print ( "X_train.shape: " , X_train.shape) 
print ( "X_valid.shape:" , X_valid.shape) 
print ( "X_test.shape:" , X_test.shape) 
print ( "y_train.shape: " , y_train.shape) 
print ( "y_valid.shape: " , y_valid.shape) 
print ( "y_test.shape: " , y_test.shape)
```

### define class name
```
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

### Visualize the sample images
```
fig = plt.figure(figsize=(5,5))
for ii in range(9):
   # sample a random image from X_train
   image_indx = np.random.choice(range(len(X_train)))
   image_random = X_train[image_indx]
   image_title = class_names[y_train[image_indx]]
    
   # put image into subplots
   imgplot = fig.add_subplot(3,3,ii+1) 
   imgplot.imshow(image_random)
   imgplot.set_title(image_title, fontsize=20)
   imgplot.axis('off')
```

### Calculate the overall accuracy of the predictions over training set and validation set
```
import time
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

images_shape = X_train.shape
img_shape = X_train.shape
n_samples = img_shape[0]
width = img_shape[1]
height = img_shape[2]
n_samples = images_shape[0]                            
X_training = X_train.reshape(n_samples, width*height)
X_training.shape

images_shape = X_test.shape
img_shape = X_test.shape
n_samples = img_shape[0]
width = img_shape[1]
height = img_shape[2]
n_samples = images_shape[0]                            
X_testing = X_test.reshape(n_samples, width*height)
X_testing.shape

images_shape = X_valid.shape
img_shape = X_valid.shape
n_samples = img_shape[0]
width = img_shape[1]
height = img_shape[2]
n_samples = images_shape[0]                             
X_validing = X_valid.reshape(n_samples, width*height)
X_validing.shape

start = time.time()
kNN_classifier = KNeighborsClassifier(n_neighbors=7)
kNN_classifier.fit(X_training, y_train)
y_test_predicted_label1 = kNN_classifier.predict(X_testing)

end = time.time()
time_duration = end-start
print("Program finishes in {} seconds in test:".format(time_duration))

accuracy_score(y_test, y_test_predicted_label1)
```
```
start = time.time()
kNN_classifier = KNeighborsClassifier(n_neighbors=3)
kNN_classifier.fit(X_training, y_train)
y_valid_predicted_label = kNN_classifier.predict(X_validing)

end = time.time()
time_duration = end-start

accuracy_score(y_valid, y_valid_predicted_label)
```

### Calculate the per-class accuracy of the predictions over training set and validation set
```
for i in range(0,10):
  y_test_img = y_test[y_test==i]
  y_test_img_pred = y_test_predicted_label1[y_test==i]
  accuracy = "{:.5f}".format(accuracy_score(y_test_img, y_test_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))
```
```
for i in range(0,10):
  y_valid_img = y_valid[y_valid==i]
  y_valid_img_pred = y_valid_predicted_label[y_valid==i]
  accuracy = "{:.5f}".format(accuracy_score(y_valid_img, y_valid_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))
```
```
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predicted_label1)
plt.title("Classification Confusion matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_valid, y_valid_predicted_label)
plt.title("Classification Confusion matrix")
plt.show()
```

### KNN
```
from sklearn.neighbors import KNeighborsClassifier as KNN

knn = KNN(n_neighbors = 7)
knn.fit(X_training,y_train)
```

### Save the trained model to disk
```
import pickle

model_filename = "My_KNN_model.sav"

saved_model = pickle.dump(knn, open(model_filename,'wb'))

print('Model is saved into to disk successfully Using Pickle')
```

### Q1: How is the softmax regression used for multi-class classification?
Softmax regression is to input a vector, and then map it to a probability vector through Softmax consensus, and finally divide it into the class with the highest probability.
### Q2: What's the cost function for softmax regression to implement the multi-class classification
The cost function of Softmax regression is gradient descent

### Calculate the overall accuracy of the predictions over training set and validation set
```
from sklearn import datasets
import numpy as np
print("Program finishes in {} seconds in valid:".format(time_duration))
```
```
from sklearn.linear_model import LogisticRegression 
start = time.time()
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10 ) 
softmax_model.fit(X_training, y_train) 
softmax_model.predict(X_validing)

end = time.time()
time_duration = end-start
print("Program finishes in {} seconds in:".format(time_duration))
```
```
from sklearn.linear_model import Perceptron
perceptron_classifier = Perceptron()
perceptron_classifier.fit(X_training, y_train)
```
```
import sklearn
print("Training accuracy", sklearn.metrics.accuracy_score(y_train, perceptron_classifier.predict(X_training)))
```
```
from sklearn.linear_model import Perceptron
perceptron_classifier = Perceptron()
perceptron_classifier.fit(X_validing, y_valid)

print("validing accuracy", sklearn.metrics.accuracy_score(y_valid, perceptron_classifier.predict(X_validing)))
```

### Calculate the per-class accuracy of the predictions over training set and validation set
```
for i in range(0,10):
  y_test_img = y_test[y_test==i]
  y_test_img_pred = perceptron_classifier.predict(X_testing)[y_test==i]
  accuracy = "{:.5f}".format(accuracy_score(y_test_img, y_test_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))
```
```
for i in range(0,10):
  y_valid_img = y_valid[y_valid==i]
  y_valid_img_pred = perceptron_classifier.predict(X_validing)[y_valid==i]
  accuracy = "{:.5f}".format(accuracy_score(y_valid_img, y_valid_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))
```
```
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, perceptron_classifier.predict(X_testing))
plt.title("Classification Confusion matrix")
plt.show()
```

### Save the trained model to disk
```
# Fit the model on training set
model = LogisticRegression()
model.fit(X_training, y_train)
# save the model to disk
filename = 'softmax_model.sav'
pickle.dump(model, open(filename, 'wb'))
```

### Creating neural network through sequential() using keras
```
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(50, activation='relu')) # input layer -> hidden layer 1
# define your loss function below
model.compile(loss='sparse_categorical_crossentropy', optimizer=('sgd'), metrics=["accuracy"])
 
model.summary()
```



```
fashion_minst = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_minst.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_train, X_valid, X_test, y_train, y_valid, y_test 
print ( "X_train.shape: " , X_train.shape) 
print ( "X_valid.shape:" , X_valid.shape) 
print ( "X_test.shape:" , X_test.shape) 
print ( "y_train.shape: " , y_train.shape) 
print ( "y_valid.shape: " , y_valid.shape) 
print ( "y_test.shape: " , y_test.shape)
```
```
history = model.fit(X_train, y_train, batch_size=5, epochs=30, validation_data=(X_valid, y_valid))
```
```
model.evaluate(X_test, y_test)
```
```
train_predictions = model.predict(X_train) 
test_predictions = model.predict(X_test) 
print("train_predictions: ",train_predictions)
```
```
X_training = X_train.reshape(55000,28*28)
X_testing = X_test.reshape(10000,28*28)
X_validing = X_valid.reshape(5000,28*28)

import sklearn
from sklearn.linear_model import Perceptron
perceptron_classifier = Perceptron()
perceptron_classifier.fit(X_training, y_train)

print("Training accuracy", sklearn.metrics.accuracy_score(y_train, perceptron_classifier.predict(X_training)))

perceptron_classifier = Perceptron()
perceptron_classifier.fit(X_validing, y_valid)

print("validing accuracy", sklearn.metrics.accuracy_score(y_valid, perceptron_classifier.predict(X_validing)))
```
```
for i in range(0,10):
  y_test_img = y_test[y_test==i]
  y_test_img_pred = perceptron_classifier.predict(X_testing)[y_test==i]
  accuracy = "{:.5f}".format(accuracy_score(y_test_img, y_test_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))
```
```
for i in range(0,10):
  y_test_img = y_test[y_test==i]
  y_test_img_pred = perceptron_classifier.predict(X_testing)[y_test==i]
  accuracy = "{:.5f}".format(accuracy_score(y_test_img, y_test_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))
```
```
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, perceptron_classifier.predict(X_testing))
plt.title("Classification Confusion matrix")
plt.show()
```
```
from tensorflow.keras.models import load_model
model.save("neural_network.m1")
```
```
#Task 4.4.1:
model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape = [28,28]))

#Dense relu

model.add(keras.layers.Dense(1000,activation = "relu"))
model.add(keras.layers.Dense(1000,activation = "relu"))
model.add(keras.layers.Dense(10,activation = "relu"))\

model.compile(loss="sparse_categorical_crossentropy", optimizer = ('sgd'), metrics = ["accuracy"])

model.summary()
```
```
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```
```
model.evaluate(X_test, y_test)
```
```
X_training = X_train.reshape(55000,28*28)
X_testing = X_test.reshape(10000,28*28)
X_validing = X_valid.reshape(5000,28*28)
```
```
#Task 4.4.2
import sklearn
from sklearn.linear_model import Perceptron
perceptron_classifier = Perceptron()
perceptron_classifier.fit(X_training, y_train)

print("Training accuracy", sklearn.metrics.accuracy_score(y_train, perceptron_classifier.predict(X_training)))

perceptron_classifier = Perceptron()
perceptron_classifier.fit(X_validing, y_valid)

print("validing accuracy", sklearn.metrics.accuracy_score(y_valid, perceptron_classifier.predict(X_validing)))
```
```
#Task 4.4.3
for i in range(0,10):
  y_test_img = y_test[y_test==i]
  y_test_img_pred = perceptron_classifier.predict(X_testing)[y_test==i]
  accuracy = "{:.5f}".format(accuracy_score(y_test_img, y_test_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))

for i in range(0,10):
  y_test_img = y_test[y_test==i]
  y_test_img_pred = perceptron_classifier.predict(X_testing)[y_test==i]
  accuracy = "{:.5f}".format(accuracy_score(y_test_img, y_test_img_pred))
  print("Accuracy of img " + str(i) +  " is: " + str(accuracy))
```
```
#Task 4.4.4
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, perceptron_classifier.predict(X_testing))
plt.title("Classification Confusion matrix")
plt.show()
```
```
#4.45
from tensorflow.keras.models import load_model
model.save("neural_network.m2")
```

### gradio
```
!pip install --quiet gradio
```
```
import gradio as gr
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import load_model

input_module1 = gr.inputs.Image(label = "test_image", image_mode='L', shape = (28,28))

input_module2 = gr.Dropdown(["KNN", "SoftMax", "NeuralNetwork_Shallow", "NeuralNetwork_Deep"])

output_module1 = gr.outputs.Textbox(label = "Predicted Class")

output_module2 = gr.Label(num_top_classes=9)
```
```
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```
```
def predict_knn(test_image):
  Knn_model = pickle.load(open('My_KNN_model.sav', 'rb'))
  predictions = Knn_model.predict_proba(test_image)
  print(predictions)
  return {class_names[i]: float(predictions[0][i]) for i in range(0,10)}
  
def predict_softmax(test_image):
  Softmax_model = pickle.load(open('softmax_model.sav', 'rb'))
  predictions = Softmax_model.predict_proba(test_image)
  return {class_names[i]: float(predictions[0][i]) for i in range(0,10)}
  
def predict_neural(test_image):
  Neural_model = load_model("neural_network.m1")
  predictions = Neural_model.predict(test_image)
  return {class_names[i]: float(predictions[0][i]) for i in range(0,10)}
  
def predict_deep_neural(test_image):
  DeepNeural_model = load_model("neural_network.m2")
  predictions = DeepNeural_model.predict(test_image)
  return {class_names[i]: float(predictions[0][i]) for i in range(0,10)}

def predictFashionClass(test_image,chosen_model):
  test_image_flatten=test_image.reshape(-1,28*28)
  if chosen_model == "KNN":
    fashion = predict_knn(test_image_flatten)
    return fashion
  elif chosen_model == "SoftMax":
    fashion = predict_softmax(test_image_flatten)
    return fashion
    
  elif chosen_model == "NeuralNetwork_Shallow":
    fashion = predict_neural(test_image_flatten)
    return fashion
    
  elif chosen_model == "NeuralNetwork_Deep":
    fashion = predict_deep_neural(test_image_flatten)
    return fashion
```
```
gr.Interface(fn=predictFashionClass,inputs=[input_module1,input_module2],outputs=[output_module1, output_module2]).launch(debug=True)
```

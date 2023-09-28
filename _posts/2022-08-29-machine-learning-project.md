## machine learning project

Part I:Apply Machine Learning approaches on MNIST Handwritten Digit Dataset
Step 1: Prepare the MNIST dataset
Requirement: the MNIST image data should be prepared and saved into the following six variables (X_train, X_valid, X_test, y_train, y_valid, y_test) with proper dimensions.
<p align="center">
  <img src="/Jerry/assets/images/step1.png" width="70%"/>
</p>


Step 2: Visualize the digit images
Requirement: randomly select 9 instances from the training set, and write codes to visualize the instances as image using matplotlib. All the nine images should be organized into 3 by 3 grids using the function 'plt.subplot()' and save in the one figure.
<p align="center">
  <img src="/Jerry/assets/images/step2.png" width="70%"/>
</p>

Step 3: Examine the frequency of classes in train, validation, and test set. 
Write codes to check the number of samples for every class in train, validation, and test set. We need to check if the data set is balanced or imbalanced dataset. You can either print out the class frequency, or visualize the class frequency.
<p align="center">
  <img src="/Jerry/assets/images/step3.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step3.2.png" width="70%"/>
</p>

Task 4: Build several classification models.  You need to fit each machine learning model on the training set (55000 images) and make predictions on the validation set (5000 images) and  test set (5000 images).
For each of the algorithms, you must write codes to answer the following steps:

Task 4.1: Proper feature scaling (Standardization  or Min-Max normalization) on the training, validation and test set

Task 4.2: Report running time of model training for all methods above.

Task 4.3: Calculate the overall accuracy of the predictions over training set, validation set and test set.

Task 4.4: Calculate the per-class accuracy of the predictions over training set, validation set and test set. For instance, among images of every class, how many of them are correctly predicted. 

Task 4.5: Visualize the classification confusion matrix on training, validation and test set to check the details of predictions over every class. 

Task 4.6: Save the trained model to disk.

(1) Softmax regression 
<p align="center">
  <img src="/Jerry/assets/images/step4.1.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.1.2.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.1.3.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.1.4.png" width="70%"/>
</p>

(2) KNearest-neighbors 
<p align="center">
  <img src="/Jerry/assets/images/step4.2.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.2.2.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.2.3.png" width="70%"/>
</p>

(3) Deep Neural Network 
<p align="center">
  <img src="/Jerry/assets/images/step4.3.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.3.2.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.3.3.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.3.4.png" width="70%"/>
</p>

(4) Support Vector Machine
<p align="center">
  <img src="/Jerry/assets/images/step4.4.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.4.2.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.4.3.png" width="70%"/>
</p>

(5) Decision Tree  
<p align="center">
  <img src="/Jerry/assets/images/step4.5.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.5.2.png" width="70%"/>
</p>

(6) Random Forest  
<p align="center">
  <img src="/Jerry/assets/images/step4.6.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.6.2.png" width="70%"/>
</p>

(7) Convolutional Neural Network  
<p align="center">
  <img src="/Jerry/assets/images/step4.7.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.7.2.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/step4.7.3.png" width="70%"/>
</p>

Task 4.7: Summarize the classification accuracy into the following table (Pandas dataframe is recommended)
<p align="center">
  <img src="/Jerry/assets/images/step4.7.png" width="70%"/>
</p>

Part II:  Deploy the machine learning models on Gradio or huggingface
<p align="center">
  <img src="/Jerry/assets/images/part2.1.png" width="70%"/>
</p>
<p align="center">
  <img src="/Jerry/assets/images/part2.2.png" width="70%"/>
</p>


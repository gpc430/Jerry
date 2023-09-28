## machine learning project

Part I:Apply Machine Learning approaches on MNIST Handwritten Digit Dataset
Step 1: Prepare the MNIST dataset
Requirement: the MNIST image data should be prepared and saved into the following six variables (X_train, X_valid, X_test, y_train, y_valid, y_test) with proper dimensions.


Step 2: Visualize the digit images
Requirement: randomly select 9 instances from the training set, and write codes to visualize the instances as image using matplotlib. All the nine images should be organized into 3 by 3 grids using the function 'plt.subplot()' and save in the one figure.

Step 3: Examine the frequency of classes in train, validation, and test set. 
Write codes to check the number of samples for every class in train, validation, and test set. We need to check if the data set is balanced or imbalanced dataset. You can either print out the class frequency, or visualize the class frequency.

Task 4: Build several classification models.  You need to fit each machine learning model on the training set (55000 images) and make predictions on the validation set (5000 images) and  test set (5000 images).
For each of the algorithms, you must write codes to answer the following steps:

Task 4.1: Proper feature scaling (Standardization  or Min-Max normalization) on the training, validation and test set

Task 4.2: Report running time of model training for all methods above.

Task 4.3: Calculate the overall accuracy of the predictions over training set, validation set and test set.

Task 4.4: Calculate the per-class accuracy of the predictions over training set, validation set and test set. For instance, among images of every class, how many of them are correctly predicted. 

Task 4.5: Visualize the classification confusion matrix on training, validation and test set to check the details of predictions over every class. 

Task 4.6: Save the trained model to disk.


(1) Softmax regression 

(2) KNearest-neighbors 

(3) Deep Neural Network 

(4) Support Vector Machine

(5) Decision Tree  

(6) Random Forest  

(7) Convolutional Neural Network  

Task 4.7: Summarize the classification accuracy into the following table (Pandas dataframe is recommended)

Part II:  Deploy the machine learning models on Gradio or huggingface


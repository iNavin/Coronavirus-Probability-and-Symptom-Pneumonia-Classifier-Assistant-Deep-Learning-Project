# Coronavirus-Probability-and-Symptom-Pneumonia-Classifier-Assistant-Deep-Learning-Project
In this we have tried to create a model with deep learning that could predict the probability of coronavirus a person is having based the symptoms that we have taken as input. and In this application we can also detect whether the affected covid19 patient is suffering from pneumonia which is severe symptom of coronavirus.

# Objective:
Under the wake of Covid19 crisis the need for doctors or specialist that are capable of diagnosing flu and coronavirus have increased drastically but due to the lack of such personnel’s different government /non-government organization are facing different issues. 

* For solving the above problem, we are trying to develop an application based on a sub domain of machine learning for detecting coronavirus in severe and 	 also at mild condition.

* For Coronavirus (mild case) we are taking few common symptoms and using those symptoms we are trying to predict our patient’s probability of having 	   coronavirus.

* For Coronavirus (severe case) we are detecting whether the patient has pneumonia, which is caused in severe cases of Covid19

# Facts:
<img width="492" alt="image" src="https://user-images.githubusercontent.com/41445769/219850129-d0f62f41-2b3d-47cb-b83f-f17fa33d8b00.png">
<img width="485" alt="image" src="https://user-images.githubusercontent.com/41445769/219850173-ded0412f-f77b-4430-b9f0-4b2f836fdd74.png">

From the above facts and figure we tried to develop a deep learning model that would take those common simple symptoms and predict a decent probability of having coronavirus, this approach also makes it easy to separate the cases of pneumonia from the cases that are caused by corona.

# Technical Details And Results:

our goal is divided into two-fold, first detection of coronavirus probability using mild symptoms and then detecting whether the patient is suffering from pneumonia viz caused in severe case of  Covid and as well as in mild case. 
For the initial analysis based on the mild symptoms we are using logistic model to model the probability of healthy or sick.The model builds a regression model to predict the probability that a given data entry belongs to the category numbered as “1”. Just like Linear regression assumes that the data follows a linear function, Logistic regression here models the data using sigmoid function.

set of parameters used are as follows:
1. Average fever: Continuous  
2. Body pain: 0/1(Binary)  
3. Age: Discrete  
4. Runny Nose: 0/1(Binary) 
5. Difficulty Breathing – categorical: -1/0/1

Suppose let us take an instance where 
Body Temperature = 98 F, Age Value = 18, Body Pain = No Pain,Runny Nose = Yes, Breathing Difficulty = Severe Difficulty.

# sample results:

![image](https://user-images.githubusercontent.com/41445769/219850361-35010447-b3cd-4f9a-8a6f-a9356dac34ea.png)

![image](https://user-images.githubusercontent.com/41445769/219850399-2e878c8a-988f-4b80-860c-bd0a0051be68.png)

For the Severe case we  are predicting the presence of Pneumonia which may have been symptom of  severe coronavirus, here we are using Convolutional neural network CNN Sequential model with Keras ,a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and is able to differentiate one from the other, as our classification models. For increasing the accuracy, we are using Dense layer to avoid overfitting , Flatten layer, Max Pooling layers , series of convolution layers with filters ,Pooling, fully connected layers (FC) and transfer leaning using inception V3 architecture

# sample results:

<img width="353" alt="image" src="https://user-images.githubusercontent.com/41445769/219850466-80ac75f8-8697-4fcb-94d0-85627f5a971f.png">

<img width="350" alt="image" src="https://user-images.githubusercontent.com/41445769/219850478-4bec7aeb-dbdb-488e-b524-828736998e82.png">

# Accuracy Metrics of Transfer Learning Architecture

<img width="222" alt="image" src="https://user-images.githubusercontent.com/41445769/219850528-eff40136-12d4-429b-bec1-ecc9245c5bed.png">
confusion matrix of CNN model

 Precision : 370 / (370 + 62) =   0.8564814814814815
 Recall : 370 / (370 + 20) = 0.9487179487179487
 F1 score : 2*0.9487*0.8564 / (0.9487 + 0.8564) = 0.9001902166084982

These are metrics produced by the Inception v3 model  





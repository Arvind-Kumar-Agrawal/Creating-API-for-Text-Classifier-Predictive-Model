# Creating-API-for-Text-Classifier-Predictive-Model
Creating API for Text Classifier Predictive Model
Text classification is the process of assigning tags or categories to text according to its content. We will be building a text classification model and create an api for serving. Let us start.

Training Data : We are using small data set related to T20 cricket leagues to understand the code flow line by line
![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/11.jpg)

Training Data Set
Code Component Details:
![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/22.jpg)

Code Components
We will go through each of these code components

trainingData.json : It contains the training data set in json format

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/33.jpg)

Training Data set in json format
G:\textClassifier\machine_learning_training_service\api\run.py

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/44.jpg)

We are using flask to run the services. In our case the training service will be running at http://localhost:9060/

G:\textClassifier\machine_learning_training_service\api\app\modelTrainer.py

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/55.jpg)

The service to initiate the training would be http://localhost:9060//api/v1.0/train/textClassifier/model

The above API will call the function textClassifierTrainer() which is inside modelTrainer.py

G:\textClassifier\machine_learning_training_service\api\app\utility\textClassificationModeler.py

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/66.jpg)

model builder code
The above code 1) reads the training file 2) converts the data to bag of words using 1 hot encoding 3) prepares training data set 4) builds a 3 layers neural network 4) saves the model files at G:\textClassifier\trainedModel

Execution Steps

Open the conda prompt and go to G:\textClassifier\machine_learning_training_service\api\

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/77.jpg)

execute python run.py

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/88.jpg)

The application is up and running at http://localhost:9060/

Open Advance Rest Client chrome extension and execute http://localhost:9060/api/v1.0/train/textClassifier/model as shown below:

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/a.jpg)

Log shows training got completed

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/b.jpg)

Model files gets generated at G:\textClassifier\trainedModel

![](https://github.com/Arvind-Kumar-Agrawal/Creating-API-for-Text-Classifier-Predictive-Model/blob/master/pics/c.jpg)

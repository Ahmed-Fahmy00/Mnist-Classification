import numpy as np
import Naive_Bayes as NB

data=np.load('mnist.npz')

xTrainRaw=data['x_train']
yTrainRaw=data['y_train']
xTestRaw=data['x_test']
yTestRaw=data['y_test']

trainFilter=(yTrainRaw==3)|(yTrainRaw==8)
testFilter=(yTestRaw==3)|(yTestRaw==8)

xTrain=xTrainRaw[trainFilter]
yTrain=yTrainRaw[trainFilter]
xTest=xTestRaw[testFilter]
yTest=yTestRaw[testFilter]

xTrain=xTrain.reshape(xTrain.shape[0],-1)
xTest=xTest.reshape(xTest.shape[0],-1)

#image flattening
xTrain=xTrain.reshape(xTrain.shape[0],-1)
xTest=xTest.reshape(xTest.shape[0],-1)

#normalization
xTrain=xTrain/255
xTest=xTest/255

model=NB.NaiveBayes()
model.fit(xTrain,yTrain)
print("Training completed")

yPred=model.predict(xTest)
accuracy=np.mean(yPred==yTest)
y_train_pred = model.predict(xTrain)
train_acc = np.mean(y_train_pred == yTrain)
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

#=numbers were misclassified
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPred))

# Precision, Recall, and F1-Score
print("\nClassification Report:")
print(classification_report(yTest, yPred))
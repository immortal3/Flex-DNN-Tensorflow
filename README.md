# Flex-DNN-Tensorflow

### Flexible Deep Neural Network based on tensorflow core api

Flex-DNN has ability to create simple Deep Neural Network for classification and regression.It's created as part of learning Tensorflow core api.

Flex-DNN can be used to do some simple task like regression and classification very fast.


Example:

Regression on Wine Quality Dataset

```python
from Data import *
from DNN import *
import numpy as np

# importing Data from excel

# EXCEL_FLAG = FALSE for reading csv file
# categorical_label = False for regression task
wineData =ExcelCSVFileData(filename="Data/winequality-red.csv",EXCEL_FLAG=False,output_label="quality",split_ratio=0.2,categorical_label=False)

X_train , Y_train = wineData.train_data()
X_test , Y_test = wineData.test_data()
```


```python
# total 4 layer NN including input and output layer
# each hidden layer contains 8 neuron
# we have 11 features like fixed acidity,volatile acidity,citric acid,residual sugar etc.
# Quality has range 0 to 10.
dnn = DNN(no_hidden_layer=2,hidden_node_list=[8,8],hidden_and_output_layer_activation_list=["sigmoid","sigmoid","relu"])
dnn.build(input_vector_shape=11,output_vector_shape=1)
dnn.loss_and_optimizer(loss="mse",optimizer="sgd",lr=0.03)
dnn.train(X_train,Y_train.reshape(Y_train.shape[0],1),batch_size=32,no_epochs=500)
```


```python
# testing our trained model
Y_pred = dnn.predict(X_test)
print ("MAE")
print (dnn.accuracy_mae(Y_test.reshape(Y_test.shape[0],1),Y_pred))
```

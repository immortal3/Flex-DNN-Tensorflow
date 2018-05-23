from Data import *
from DNN import *
import numpy as np

# importing Data from excel


# EXCEL_FLAG = FALSE for reading csv file
# categorical_label = False for regression task
wineData =ExcelCSVFileData(filename="Data/winequality-red.csv",EXCEL_FLAG=False,output_label="quality",split_ratio=0.2,categorical_label=False)

X_train , Y_train = wineData.train_data()
X_test , Y_test = wineData.test_data()

# printing data
print (X_train[0:2])
print (Y_train[0:2])


# printing shape
print (X_train.shape)
print (Y_train.shape)



# total 4 layer NN including input and output layer
# each hidden layer contains 8 neuron
dnn = DNN(no_hidden_layer=2,hidden_node_list=[8,8],hidden_and_output_layer_activation_list=["sigmoid","sigmoid","relu"])
dnn.build(input_vector_shape=11,output_vector_shape=1)
dnn.loss_and_optimizer(loss="mse",optimizer="sgd",lr=0.03)
dnn.train(X_train,Y_train.reshape(Y_train.shape[0],1),batch_size=32,no_epochs=500)


# testing our trained model
Y_pred = dnn.predict(X_test)
print ("MAE")
print (dnn.accuracy_mae(Y_test.reshape(Y_test.shape[0],1).astype(np.float32),Y_pred))



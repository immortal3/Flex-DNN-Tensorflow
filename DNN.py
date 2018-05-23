from tensorflow_utils import *
from tqdm import tqdm
import numpy as np



class DNN:


	def __init__(self,no_hidden_layer,hidden_node_list,hidden_and_output_layer_activation_list):
		self.no_hidden_layer = no_hidden_layer
		self.hidden_and_output_layer_activation_list = hidden_and_output_layer_activation_list
		self.hidden_node_list = []
		for i in hidden_node_list:
			self.hidden_node_list.append(i)
		
		assert ( self.no_hidden_layer == len(self.hidden_node_list),"Lenght Hidden Node list is not matching with Number of Hidden Nodes")
		assert ( self.no_hidden_layer + 1 == len(self.hidden_and_output_layer_activation_list) , "Mis Match")
		self.model_tesors = {}



	def build(self,input_vector_shape,output_vector_shape):
		self.model_tesors['input'] = tf.placeholder(shape=[None,input_vector_shape],dtype=tf.float32,name="input")
		self.model_tesors['output'] = tf.placeholder(shape=[None,output_vector_shape],dtype=tf.float32,name="output")

		base_name = "hidden_"

		prev_tensor = self.model_tesors['input']
		for cnt,i in enumerate(self.hidden_node_list):
			tesor_name = base_name+str(cnt)
			self.model_tesors[tesor_name] = full_layer(input_vector=prev_tensor,size=i,prefix_name=tesor_name,activation=self.hidden_and_output_layer_activation_list[cnt])
			prev_tensor = self.model_tesors[tesor_name]

		self.model_tesors['output_layer'] = full_layer(input_vector=prev_tensor,size=output_vector_shape,prefix_name="output_layer",activation="relu")


	def loss_and_optimizer(self,loss="mse",optimizer="adam",lr=0.001):
		self.lr = lr
		if loss == "mse":
			self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.model_tesors['output'],self.model_tesors['output_layer']))
		elif loss == "binary_cross":
			self.loss = tf.keras.losses.binary_crossentropy(self.model_tesors['output'],self.model_tesors['output_layer'])
		elif loss == "one-hot":
			self.loss = tf.losses.softmax_cross_entropy(self.model_tesors['output'],self.model_tesors['output_layer'])
		if optimizer == "adam":
			self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		elif optimizer == "sgd":
			self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)


	def train(self,X,Y,batch_size=32,no_epochs=5,save_path_model="/tmp/model.ckpt"):
		self.saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(no_epochs):
				for i in tqdm(range(0,X.shape[0],32)):
					temp_loss,_ = sess.run((self.loss,self.optimizer),feed_dict={self.model_tesors['input']:X,self
						.model_tesors['output']:Y})
				print ("loss:",temp_loss)
			save_path = self.saver.save(sess,save_path_model)


	def predict(self,X,save_path_model="/tmp/model.ckpt"):
		with tf.Session() as sess:
			self.saver.restore(sess,save_path_model)
			temp_Y = sess.run(self.model_tesors['output_layer'],feed_dict={self.model_tesors['input']:X})
		return temp_Y

	def accuracy_mae(self,y_true,y_pred):
		if self.hidden_and_output_layer_activation_list[self.no_hidden_layer] == "None" or self.hidden_and_output_layer_activation_list[self.no_hidden_layer] == "relu":
			mae = np.sum(abs(y_true - y_pred))
			mae = mae / y_true.shape[0]
			return mae
		else:
			print ("Not possible")


# if __name__ == "__main__":
# 	dnn = DNN(1,[1],["None","None"])
# 	dnn.build(1,1)
# 	dnn.loss_and_optimizer(lr=0.1)
# 	x = np.arange(100)
# 	delta = np.random.uniform(-2,2, size=(100,))
# 	y = 4*x + 3 + delta
# 	dnn.train(x.reshape(100,1),y.reshape(100,1),batch_size=1,no_epochs=30)
# 	print (dnn.predict(np.array([3]).reshape(1,1)))





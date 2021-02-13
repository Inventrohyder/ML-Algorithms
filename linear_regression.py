import streamlit as st
import pandas as pd
import random
from mxnet import autograd, np, npx
npx.set_np()

#Generate the synthetic dataset (standard deviation = 0.01)
def synthetic_data(w, b, num_examples, mean, sd):
		"""Generate y = Xw + b + noise."""
		X = np.random.normal(0, 1, (num_examples, len(w)))
		y = np.dot(X, w) + b
		y += np.random.normal(mean, sd, y.shape)
		return X, y.reshape((-1, 1))
		
#Define the model - linear regression model
def linreg(X, w, b):
		"""The linear regression model."""
		#Broadcasting since Xw is a vector and b is a scalar
		return np.dot(X, w) + b

#Defining the loss function
def squared_loss(y_hat, y):
		"""Squared loss."""
		return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
		
#Define the optimization algorithm
def sgd(params, lr, batch_size):
		"""Minibatch stochastic gradient descent."""
		for param in params:
				param[:] = param - lr * param.grad / batch_size
				
#Generate minbatches for the training data
def data_iter(batch_size, features, labels):
		"""
		Objective: The function yields minibatches of the size batch_size. 
		Input: int, list, list: a batch size, a matrix of features, and a vector of
		labels
		Output: tuple: a group of features and labels.
		"""
		num_examples = len(features)
		indices = list(range(num_examples))
		random.shuffle(indices)
		for i in range(0, num_examples, batch_size):
				batch_indices = np.array(
						indices[i: min(i + batch_size, num_examples)])
				yield features[batch_indices], labels[batch_indices]
				
if __name__ == '__main__':
	#initialize weights by sampling random numbers from a normal distribution
	st.sidebar.header('Data Generation')
	data_size = st.sidebar.selectbox('Size of data', [1000, 5000, 10000])
	mean = st.sidebar.slider('Mean', 0.0, 1.0, 0.0)
	sd = st.sidebar.slider('Standard deviation', 0.01, 0.05, 0.01)
	
	st.sidebar.header('Hyperparameter tuning')
	lr = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.03)
	num_epochs = st.sidebar.slider('Number of epochs', 1, 10, 5)	
	batch_size = st.sidebar.slider('Batch size', 5, 20, 10)			
	
	w = np.random.normal(mean, sd, (2, 1))
	b = np.zeros(1)
	w.attach_grad()
	b.attach_grad()
					
	#Define network and loss function
	net = linreg
	loss = squared_loss
	    
	#assign true features, w and the generated features and labels
	#random.seed(123)
	true_w = np.array([2, -3.4])
	true_b = 4.2
	features, labels = synthetic_data(true_w, true_b, data_size, mean, sd)
	#labels = np.array(labels.flatten())
	#column_values = np.array([features[:, 0], features[:, 1],labels]).reshape(1000, 3)
	#column_values = np2.array(column_values).shape
	#df = pd.DataFrame(column_values, columns=['Feature 1', 'Feature 2', 'Labels'])
	#df_display = df.head()

	for X, y in data_iter(batch_size, features, labels):
			print(X, '\n', y)
			break

	st.write("""# Retrieving Linear Regression Parameters from Scratch""")
	st.markdown("""
	The program below attempts to retrieve the true parameters from a randomly generated dataset using linear regression 
	* **Python libraries:** mxnet, pandas, streamlit, random
	""")
	
	st.text("Example Feature (w): {}".format(features[0]))
	st.text("Example Label (b): {}".format(labels[0]))

	st.text("True Feature (w): {}".format(true_w))
	st.text("True Label (b): {}".format(true_b))

	st.write("""### First 5 examples of the dataset""")
	#st.dataframe(df_display)

	for epoch in range(num_epochs):
			for X, y in data_iter(batch_size, features, labels):
				with autograd.record():
					l = loss(net(X, w, b), y) # Minibatch loss in `X` and `y`
					l.backward()
				sgd([w, b], lr, batch_size) # Update parameters using their gradient
			train_l = loss(net(features, w, b), labels)
			st.text(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
			
	st.text("Predicted Feature (w): {}".format(w.flatten()))
	st.text("Predicted Label (b): {}".format(b))

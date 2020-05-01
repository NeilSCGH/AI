import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from lib.tools import *
import math


class program():
	def __init__(self,args):
	    self.tool = tools(args)

		#the column to predict
	    #self.targetColumn = "Overall"
	    #self.fileName = "data_cleaned.csv"
	    self.setup(args)

	def setup(self,args):
	    if self.tool.argHasValue("-f"):#the file where to read data
	      val=self.tool.argValue("-f")
	      self.fileName=val
	    else:
	      self.stop("Error, -f is missing !")

	    if self.tool.argHasValue("-o"):#the column to predict
	      val=self.tool.argValue("-o")
	      self.targetColumn=val
	    else:
	      self.stop("Error, -o is missing !")

	def run(self):
		##Getting the cleaned data
		try:
			df_clean = pd.read_csv(self.fileName)
		except:
			self.stop("Error, file not found !")

		y = df_clean[self.targetColumn] #result column
		X = df_clean.drop(self.targetColumn, axis=1, inplace=False)
		#y.hist(bins=100)

		#Spliting the data set into a training set and a test set
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
		print("train set shape: ", X_train.shape, y_train.shape)
		print("test set shape: ", X_test.shape, y_test.shape)

		##Training the model
		#Fitting a linear model to the training set
		reg = LinearRegression()
		reg.fit(X_train, y_train)

		train_score = reg.score(X_train, y_train)
		print  ('\ntrain score =', np.round(train_score,6))

		##Testing the model
		y_pred = reg.predict(X_test)
		test_score = r2_score(y_test,y_pred)
		print  ('test score =', np.round(test_score,6))

		mse = mean_squared_error(y_test, y_pred)
		mae = mean_absolute_error(y_test, y_pred)
		print ('mse = {}, rmse = {} mae = {}'.format(np.round(mse,6),np.round(math.sqrt(mse),6),np.round(mae,6)))

		#Plotting a scatter plot where y_test is in the x axis and  y_pred is in the y axis
		plt.figure(figsize= (10, 5))
		plt.scatter(y_test, y_pred)
		plt.xlabel('y_test')
		plt.ylabel('y_pred')
		#plt.show()

	def stop(self,msg):
		print(msg)
		exit(0)
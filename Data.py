from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

class FileData(ABC):

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def train_data(self):
		pass

	@abstractmethod
	def test_data(self):
		pass


class ExcelCSVFileData(FileData):

	def __init__(self,filename,output_label,EXCEL_FLAG=True,csv_sep=";",split_ratio=0.2,categorical_cols=[],categorical_label=False):
		if EXCEL_FLAG:
			self.df = pd.read_excel(filename)
		else:
			self.df = pd.read_csv(filename,sep=";")
		# print (self.df)	
		self.Y = self.df[output_label]
		self.X = self.df.drop(output_label,axis=1)
		
		for i in categorical_cols:
			self.X = pd.concat((self.X,pd.get_dummies(self.X[i], drop_first=True,prefix=[i])),axis=1)
			self.X.drop([i],axis=1,inplace=True)
		if categorical_label:
			self.Y = pd.get_dummies(self.Y,drop_first=False,prefix=[i])

		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X.as_matrix(), self.Y.as_matrix(), test_size=split_ratio, random_state=42)
	
	def train_data(self):
		return self.x_train,self.y_train

	def test_data(self):
		return self.x_test,self.y_test

if __name__ == "__main__":
	eData = ExcelFileData(filename="temp.xlsx",output_label="label",categorical_cols=["b2"])
	x,y = eData.test_data()
	print (x,y)

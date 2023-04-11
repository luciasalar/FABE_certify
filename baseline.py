#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import os

from ruamel import yaml #for parameters setting
import datetime
import csv
import gc
import glob

#This classifier predicts an outcome whether a person will have income > 50k
class Process_data:
	def __init__(self):
		self.path = '/Users/luciachen/Desktop/Desktop_MacbookPro/Fabe_certify/'

	def read_file(self) -> pd.DataFrame:
		""" Read dataset """

		adult_raw = pd.read_csv(self.path + 'adult.csv') 
		return adult_raw


	def create_dummies(self):
		#subset variables to create dummies

		adult_dummy = adult_raw[['workclass', 'education', 'marital-status', 'occupation', 'relationship','race', 'gender', 'income']]

		# recode some of the variables with multiple categories into 2 categories
		adult_dummy['race'] = adult_dummy['race'].replace(to_replace = ['Asian-Pac-Islander', 'Black', 'Other', 'Amer-Indian-Eskimo'], value = 'nonWhite', regex=True)
		
		adult_dummy['education'] = adult_dummy['education'].replace(to_replace = ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Preschool'], value = 'below_Bachelors')
		adult_dummy['education'] = adult_dummy['education'].replace(to_replace = ['Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Some-college', 'Prof-school', 'Assoc-acdm', 'Assoc-voc'], value = 'above_Bachelors')
		print(adult_dummy['education'].value_counts())

		dummies = pd.get_dummies(adult_dummy, prefix=['workclass', 'education', 'marital-status', 'occupation', 'relationship','race', 'gender', 'income'],  drop_first=True)

		# combine with other variables, lacking of native-country
		non_dummies  = adult_raw[['age', 'fnlwgt', 'educational-num',  'capital-gain', 'capital-loss', 'hours-per-week']]
		all_col =  pd.concat([dummies, non_dummies], axis=1)
		all_col.rename(columns={"income_>50K": "income"}, inplace= True) #change the column 
		
		return all_col

	def get_train_test_split(self) -> pd.DataFrame:
		'''get train test set'''

		all_col = self.create_dummies()
		y  = all_col["income"]
		X = all_col.drop(["income"], axis=1)

		# get 20% holdout set for testing
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 300)

		return X_train, X_test, y_train, y_test


class ColumnSelector:
	'''feature selector for pipline (pandas df format) '''
	def __init__(self, columns):
		self.columns = columns

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		assert isinstance(X, pd.DataFrame)

		try:
			return X[self.columns]
		except KeyError:
			cols_error = list(set(self.columns) - set(X.columns))
			raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

	def get_feature_names(self):
		return self.columns.tolist



class Training:
	def __init__(self,  X_train, X_test, y_train, y_test, parameters, features_list):
		self.path = '/Users/luciachen/Desktop/Desktop_MacbookPro/Fabe_certify/'
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.parameters = parameters
		self.features_list = features_list

 
	def get_feature_by_names(self):
		"""Select all the merged features. """

		fea_list = []
		for fea in self.features_list: #here we loop through each column name
			f_list = [i for i in self.X_train.columns if fea in i]  #select column with keywords defined in feature list on parameters.yaml
			fea_list.append(f_list)
		#flatten feature list
		flat = [x for sublist in fea_list for x in sublist]
		
		return flat


	def setup_pipeline(self, classifier):
		'''set up pipeline'''
		features_col = self.get_feature_by_names()

		pipeline = Pipeline([
		# ColumnSelector(columns = features_list),
			
			('feats', FeatureUnion([
	   
		  # feature sets are defines in the yaml file

				('other_features', Pipeline([

					('selector', ColumnSelector(columns=features_col)),
					('impute', SimpleImputer(strategy='mean')),# impute nan with mean
				])),

			 ])),

			   ('clf', Pipeline([
			   ('scale', StandardScaler(with_mean=False)),  # scale features
				('classifier', classifier),  # classifier defined in parameters.yaml
		   
				 ])),
		])
		return pipeline

	

	def training_models(self, pipeline):
		'''train models with grid search'''
		grid_search_item = GridSearchCV(pipeline, self.parameters, cv=5, scoring='accuracy') #select best model by accuracy
		grid_search = grid_search_item.fit(self.X_train, self.y_train)
		
		return grid_search


	def evaluation_methods(self, y_true, y_pred):
		"""compute TPR, FPR, PPV """

		#get confusion matrix for group 
		CM = confusion_matrix(y_true, y_pred)
		TN = CM[0][0]
		FN = CM[1][0]
		TP = CM[1][1]
		FP = CM[0][1]

		#calculate recall: TPR
		TPR = TP / (TP + FN)

		#calculate FPR
		FPR = FP / (FP + TN)

		#calculate PPV
		PPV = TP / (TP + FP)

		return TPR, FPR, PPV

	def get_group_evaluation(self, group_name):
		"""compare performances in different groups
		group_name: groups for comparison

		"""


		X_1 = X_test.loc[X_test[group_name] == 1]
		X_0 = X_test.loc[X_test[group_name] == 0]

		#merge feature set with outcome in test set
		all_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
		
		#subset outcome according to group name
		y_1 = all_test.loc[all_test[group_name] == 1].income
		y_0 = all_test.loc[all_test[group_name] == 0].income


		#used the trained model to make predictions on different groups, y_pred_prob_1: prediciton, y_true_1: outcome
		y_true_1, y_pred_1 = y_1, grid_search.predict(X_1)
		y_true_0, y_pred_0 = y_0, grid_search.predict(X_0)

		#get confusion matrix for group 
		TPR_1, FPR_1, PPV_1 = self.evaluation_methods(y_true_1, y_pred_1)
		TPR_0, FPR_0, PPV_0 = self.evaluation_methods(y_true_0, y_pred_0)

		return TPR_1, TPR_0, FPR_1, FPR_0, PPV_1, PPV_0 



	def test_model(self, classifier):
		'''train model, use model to predict on new data and save results'''
	   
		#training model
		print('getting pipeline...')

		#the dictionary returns a list, here we extract the string from list use [0]
		pipeline = self.setup_pipeline(eval(classifier)())

		# train model
		print('features', self.features_list)
		grid_search = self.training_models(pipeline)
		# make prediction on test set
		print('prediction...')
	  
		y_true, y_pred = self.y_test, grid_search.predict(self.X_test)

		# store classification report
		report = classification_report(y_true, y_pred, digits=2)

		# store prediction result
		y_pred_series = pd.DataFrame(y_pred)
		result = pd.concat([y_true.reset_index(drop=True), y_pred_series, self.X_test['post_id'].reset_index(drop=True)], axis = 1)
		result.columns = ['y_true', 'y_pred', 'post_id']

	 
		return report, grid_search, pipeline
   


def load_experiment(path_to_experiment):
	#load experiment 
	data = yaml.safe_load(open(path_to_experiment))
	return data





p = Process_data()
adult_raw = p.read_file()
all_col = p.create_dummies()
experiment = load_experiment(p.path + 'parameters.yaml')
X_train, X_test, y_train, y_test = p.get_train_test_split()


#store results
file_exists = os.path.isfile(p.path + 'results/test_result.csv') #remember to create a results folder
f = open(p.path + 'results/test_result.csv', 'a')
writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

#if result file doesn't exist, create file and write column names
if not file_exists:
	writer_top.writerow(['best_scores'] + ['best_parameters'] + ['report']  + ['time'] + ['model'] +['feature_set'] + ['TPR_male'] + ['TPR_female'] + ['FPR_male'] + ['FPR_female'] + ['PPV_male'] + ['PPV_female'] + ['TPR_nonWhite'] + ['TPR_White'] +  ['FPR_nonWhite'] +  ['FPR_White'] + ['PPV_nonWhite'] +  ['PPV_White'] + ['TPR_below_B'] + ['TPR_above_B'] +  ['FPR_below_B'] +  ['FPR_above_B'] +  ['PPV_below_B'] + ['PPV_above_B'])
	f.close()

# loop through each classifier and parameters defined in experiment.yaml, then we get the classification report of general performance andfe compare the log loss in each group defined by sensitive attributes

for classifier in experiment['experiment']:
	parameters = experiment['experiment'][classifier]
	
	#loop through lists of features
	for key, features_list in experiment['features'].items():
		print(features_list)
		
		#train model 
		train = Training(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, parameters=parameters, features_list=features_list)
		
		#set up pipeline
		pipeline = train.setup_pipeline(eval(classifier)())
		
		#grid search
		grid_search = train.training_models(pipeline)

		#get predictions
		y_true, y_pred = y_test, grid_search.predict(X_test)

		#get classification report
		report = classification_report(y_true, y_pred, digits=2)
		
# # 		# sensitive attributes 
		TPR_male, TPR_female, FPR_male, FPR_female, PPV_female, PPV_male  = train.get_group_evaluation('gender_Male') 
		TPR_nonWhite, TPR_White, FPR_nonWhite, FPR_White, PPV_nonWhite, PPV_White  = train.get_group_evaluation('race_nonWhite') 
		TPR_below_B, TPR_above_B, FPR_below_B, FPR_above_B, PPV_below_B, PPV_above_B  = train.get_group_evaluation('education_below_Bachelors') 


		# combine the result columns: grid search best score, best parameters, classification report, log loss, experiment time, classifier, feature set, log loss of sensitive groups
		result_row = [[grid_search.best_score_, grid_search.best_params_, report, str(datetime.datetime.now()), classifier, features_list,  TPR_male, TPR_female, FPR_male, FPR_female, PPV_female, PPV_male, TPR_nonWhite, TPR_White, FPR_nonWhite, FPR_White, PPV_nonWhite, PPV_White, TPR_below_B, TPR_above_B, FPR_below_B, FPR_above_B, PPV_below_B, PPV_above_B]]

		# store test result
		f = open(p.path + 'results/test_result.csv', 'a')
		writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

		writer_top.writerows(result_row)

		f.close()
		gc.collect()
















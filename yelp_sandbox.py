# Run some recommendation experiments using Yelp
import sys
import csv
import json

import pandas
import numpy as np 
import scipy.sparse
import matplotlib.pyplot as plt

def load_businesses(root):
	'''
	Load attributes for all 144,000 businesses

	Columns:
	['address' 'attributes' 'business_id' 'categories' 'city' 'hours' 'is_open'
	 'latitude' 'longitude' 'name' 'neighborhood' 'postal_code' 'review_count'
	 'stars' 'state' 'type']
	'''
	file = 'yelp_academic_dataset_business'
	df = pandas.read_json('{}/{}.json'.format(root,file), lines=True)
	return df

def load_users(root):
	'''Load attributes for all users'''
	file = 'yelp_academic_dataset_user'
	df = pandas.read_json('{}/{}.json'.format(root,file), lines=True)
	return df

def load_tips(root):
	'''
	Load attributes for all tips

	Columns:
	['business_id' 'date' 'likes' 'text' 'type' 'user_id']
	'''
	file = 'yelp_academic_dataset_tip'
	df = pandas.read_json('{}/{}.json'.format(root,file), lines=True)
	return df

def load_checkins(root):
	'''
	Load attributes for all users

	Columns:
	['business_id' 'time' 'type']
	'''
	file = 'yelp_academic_dataset_checkin'
	df = pandas.read_json('{}/{}.json'.format(root,file), lines=True)
	return df

def load_review(root):
	'''Load attributes for all reviews'''
	file = 'yelp_academic_dataset_user'
	df = pandas.read_json('{}/{}.json'.format(root,file), lines=True)
	return df


if __name__ == "__main__":

	# Specify data root directory
	data_root = './data/yelp'

	# Dataset files
	file = 'yelp_academic_dataset_user'

	df_biz = load_businesses(data_root)
	print(df_biz.columns.values)
	df_tip = load_tips(data_root)
	print(df_tip.columns.values)
	df_checkin = load_checkins(data_root)
	print(df_checkin.columns.values)

	# # These are big.
	# df_user = load_users(data_root)
	# print(df_user.columns.values)
	# df_review = load_reviews(data_root)
	# print(df_review.columns.values)

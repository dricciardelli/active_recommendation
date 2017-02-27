
# Run some recommendation experiments using MovieLens 100K
import pandas
import numpy as np 
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

def euclidean_distance(v1, v2):
	return np.linalg.norm(v1 - v2)

def greedy_select(X, trials): 
	'''
	Greedily select points according to which is 
	closest to any previously seen sample

	Distance Metric: Euclidean 
	'''

	k = 2 # For visualization
	n = X_train.shape[0]

	U, s, Vt = np.linalg.svd(X_train, full_matrices=False)

	X_svd = U[:, 0:k].dot(np.diag(s[0:k]))

	# Select first point (def not random)
	seen = [0]
	for _ in range(trials):

		# Looking for point with smallest minimum distance
		min_seen = [float('inf'), -1]
		for i in [x for x in range(n) if x not in seen]:
			for j in seen:
				v1 = X_svd[i,:]
				v2 = X_svd[j,:]
				#dist = 1.0 - cosine_similarity(v1, v2)[0][0]
				#dist = cosine_similarity(v1, v2)[0][0]
				dist = euclidean_distance(v1, v2)
				if dist < min_seen[0]:
					min_seen = [dist, i]

		seen.append(min_seen[1])
		
		plt.scatter(X_svd[:,0], X_svd[:,1], c=categories, alpha=0.01)
		plt.scatter(X_svd[seen,0], X_svd[seen,1], s=100,c=[0.0]*len(seen), alpha=0.7)
		plt.show()

	print(seen)

	return

def antigreedy_select(X, trials):
	'''
	Select points according to which is 
	furthest from any previously seen sample

	Distance Metric: Euclidean
	'''

	k = 2 # For visualization
	n = X_train.shape[0]

	U, s, Vt = np.linalg.svd(X_train, full_matrices=False)

	X_svd = U[:, 0:k].dot(np.diag(s[0:k]))

	# Select first point (def not random)
	seen = [0]
	for _ in range(trials):

		# Looking for point with largest minimum distance
		max_min_seen = [float('-inf'), -1]

		# For each potential sample
		for i in [x for x in range(n) if x not in seen]:

			# Calculate the minimum distance
			min_seen = [float('inf'), -1]

			# To all previously seen samples
			for j in seen:
				v1 = X_svd[i,:]
				v2 = X_svd[j,:]
				#dist = 1.0 - cosine_similarity(v1, v2)[0][0]
				#dist = cosine_similarity(v1, v2)[0][0]
				dist = euclidean_distance(v1, v2)
				if dist < min_seen[0]:
					min_seen = [dist, i]

			if min_seen[0] > max_min_seen[0]:
				max_min_seen = min_seen

		seen.append(max_min_seen[1])
		
		plt.scatter(X_svd[:,0], X_svd[:,1], c=categories, alpha=0.01)
		plt.scatter(X_svd[seen,0], X_svd[seen,1], s=100,c=[0.0]*len(seen), alpha=0.7)
		plt.show()

	print(seen)

	return


if __name__ == "__main__":
	data_dir = "data/ml-100k/"

	df = pandas.read_csv(data_dir + "u.item", sep="|", header=-1, encoding='latin-1')
	values = df.values

	# Maintain genre data, movie_title
	X_titles = values[:,1]
	X_train = np.asfarray(values[:, 5:])

	row_sums = X_train.sum(axis=1)
	X_train = X_train / row_sums[:, np.newaxis]

	categories = [np.argmax(X_train[i,:]) for i in range(X_train.shape[0])]

	#greedy_select(X_train, trials=10)
	antigreedy_select(X_train, trials=10)



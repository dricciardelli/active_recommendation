
# Run some recommendation experiments using MovieLens 100K
import pandas
import numpy as np 
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def greedy_select(user_train, user_test, items, trials=10):

    # Find positively rated samples
    liked = [i for i in user_train.nonzero()[0] if user_train[i] > 0]
    disliked = [i for i in user_train.nonzero()[0] if user_train[i] < 0]
    unseen = user_test.nonzero()[0]

    new_samples = []

    for _ in range(trials):

        # Looking for point with smallest minimum distance
        min_seen = [float('inf'), -1]
        for i in [x for x in unseen if (x not in new_samples)]:
            for j in (liked+new_samples):
                v1 = items[i,:]
                v2 = items[j,:]
                dist = euclidean_distance(v1, v2)
                if dist < min_seen[0]:
                    min_seen = [dist, i]

        new_samples.append(min_seen[1])
        
        # All items
        plt.scatter(items[:,0], items[:,1], alpha=0.2)

        plt.scatter(items[liked,0], items[liked,1], s=50,c=['green']*len(liked), alpha=0.5)
        plt.scatter(items[disliked,0], items[disliked,1], s=50,c=['red']*len(disliked), alpha=0.5)
        plt.scatter(items[new_samples,0], items[new_samples,1], s=100,c=['blue']*len(new_samples), alpha=0.7)
        plt.show()


def antigreedy_select(user_train, user_test, items, trials=10):
    '''
    Select points according to which is 
    furthest from any previously seen sample

    Distance Metric: Euclidean
    '''

    # Find positively rated samples
    liked = [i for i in user_train.nonzero()[0] if user_train[i] > 0]
    disliked = [i for i in user_train.nonzero()[0] if user_train[i] < 0]
    unseen = user_test.nonzero()[0]

    new_samples = []

    for _ in range(trials):

        # Looking for point with largest minimum distance
        max_min_seen = [float('-inf'), -1]

        # For each potential sample
        for i in [x for x in unseen if x not in new_samples]:

            # Calculate the minimum distance
            min_seen = [float('inf'), -1]

            # To all previously seen samples
            for j in (liked+new_samples):
                v1 = items[i,:]
                v2 = items[j,:]
                #dist = 1.0 - cosine_similarity(v1, v2)[0][0]
                #dist = cosine_similarity(v1, v2)[0][0]
                dist = euclidean_distance(v1, v2)
                if dist < min_seen[0]:
                    min_seen = [dist, i]

            if min_seen[0] > max_min_seen[0]:
                max_min_seen = min_seen

        new_samples.append(max_min_seen[1])
        
        # All items
        plt.scatter(items[:,0], items[:,1], alpha=0.2)

        plt.scatter(items[liked,0], items[liked,1], s=50,c=['green']*len(liked), alpha=0.5)
        plt.scatter(items[disliked,0], items[disliked,1], s=50,c=['red']*len(disliked), alpha=0.5)
        plt.scatter(items[new_samples,0], items[new_samples,1], s=100,c=['blue']*len(new_samples), alpha=0.7)
        plt.show()

    return

data_dir = "data/ml-100k/"

# 100,000 ratings for 1000 users on 1700 movies
# Density 0.06
# 90,570 x 4: (uid, mid, rating, rid)
data_shape = (943, 1682)

df = pandas.read_csv(data_dir + "ua.base", sep="\t", header=-1)
values = df.values
values[:, 0:2] -= 1
X_train = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=np.float, shape=data_shape)

df = pandas.read_csv(data_dir + "ua.test", sep="\t", header=-1)
values = df.values
values[:, 0:2] -= 1
X_test = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=np.float, shape=data_shape)

# Compute means of nonzero elements
X_row_mean = np.zeros(data_shape[0])
X_row_sum = np.zeros(data_shape[0])

train_rows, train_cols = X_train.nonzero()

# Iterate through nonzero elements to compute sums and counts of rows elements
for i in range(train_rows.shape[0]):
    X_row_mean[train_rows[i]] += X_train[train_rows[i], train_cols[i]]
    X_row_sum[train_rows[i]] += 1

# Note that (X_row_sum == 0) is required to prevent divide by zero
X_row_mean /= X_row_sum + (X_row_sum == 0)

# Subtract mean rating for each user
for i in range(train_rows.shape[0]):
    X_train[train_rows[i], train_cols[i]] -= X_row_mean[train_rows[i]]

test_rows, test_cols = X_test.nonzero()
for i in range(test_rows.shape[0]):
    X_test[test_rows[i], test_cols[i]] -= X_row_mean[test_rows[i]]

X_train = np.array(X_train.toarray())
X_test = np.array(X_test.toarray())

#ks = np.arange(2, 50)
k = 2
train_scores = X_train[(train_rows, train_cols)]
test_scores = X_test[(test_rows, test_cols)]

# Now take SVD of X_train
# (n x k) . (k x k) . (k x d)
U, s, Vt = np.linalg.svd(X_train, full_matrices=False)

# What we really want are movie vectors (d x k)
X_svd = np.transpose(np.diag(s[0:k]).dot(Vt[0:k, :]))

# Peek at explained variance
#plt.plot(range(s.shape[0]), np.cumsum(s) / np.sum(s))
#plt.show()

# Peek at movie representations (2-d only)
plt.scatter(X_svd[:,0], X_svd[:,1])
plt.show()

# Check user rating counts
train_counts = [(i, np.count_nonzero(X_train[i,:])) for i in range(X_train.shape[0])]
#print(train_counts)

# Check which test users are good to test on.
test_counts = [(i, np.count_nonzero(X_test[i,:])) for i in range(X_test.shape[0])]
#print(test_counts)

# So there is overlap from train / test in (uid, rid, rating) tuples. 
# Every user in test is in train, with some nonzero number of ratings. 

# Some user counts:
# User 0: 262
# User 1: 52
# User 2: 44

USER = 0

#greedy_select(X_train[USER,:], X_test[USER,:], X_svd, trials=10)
antigreedy_select(X_train[USER,:], X_test[USER,:], X_svd, trials=10)



# Run some recommendation experiments using MovieLens 100K
import pandas
import numpy as np 
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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
train_mae = 0 #np.zeros(ks.shape[0])
test_mae = 0 #np.zeros(ks.shape[0])
train_scores = X_train[(train_rows, train_cols)]
test_scores = X_test[(test_rows, test_cols)]

# Now take SVD of X_train
# (n x k) . (k x k) . (k x d)
U, s, Vt = np.linalg.svd(X_train, full_matrices=False)

#for j, k in enumerate(ks):
X_pred = U[:, 0:k].dot(np.diag(s[0:k])).dot(Vt[0:k, :])

pred_train_scores = X_pred[(train_rows, train_cols)]
pred_test_scores = X_pred[(test_rows, test_cols)]

train_mae = mean_absolute_error(train_scores, pred_train_scores)
test_mae = mean_absolute_error(test_scores, pred_test_scores)

print(k,  train_mae, test_mae)

# What we really want are movie vectors (d x k)
X_svd = np.transpose(np.diag(s[0:k]).dot(Vt[0:k, :]))

# Peek at explained variance
#plt.plot(range(s.shape[0]), np.cumsum(s) / np.sum(s))
#plt.show()

# Peek at movie representations (2-d only)
plt.scatter(X_svd[:,0], X_svd[:,1])
plt.show()

# Check user rating counts
train_counts = [np.count_nonzero(X_train[i,:]) for i in range(X_train.shape[0])]

# Check which test users are good to test on.
test_counts = [(i, np.count_nonzero(X_test[i,:])) for i in range(X_test.shape[0])]
print(test_counts)
good_test_users = [x[0] for x in test_counts if x[1] > 100]
print(good_test_users)

# plt.plot(ks, train_mae, 'k', label="Train")
# plt.plot(ks, test_mae, 'r', label="Test")
# plt.xlabel("k")
# plt.ylabel("MAE")
# plt.legend()
# plt.show()

# Run some recommendation experiments using MovieLens 100K
import sys

import pandas
import numpy as np 
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def map_labels(p):
    if p > 0:
        return 'green'
    else:
        return 'red'

def plot_mesh(X, clf):
    h = 0.02 # Step size in the mesh

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn, alpha=0.1)

def greedy_select(user_train, user_test, items, trials=10, num_test=50):

    # Find positively rated samples
    liked = [i for i in user_train.nonzero()[0] if user_train[i] > 0]
    disliked = [i for i in user_train.nonzero()[0] if user_train[i] < 0]
    unseen = user_test.nonzero()[0]

    # Reserve first 50 samples in test as actual test samples
    test_unseen = unseen[:num_test]
    test_err = []
    train_err = []

    # Maintain static test set
    X_test = items[test_unseen,:]
    y_test = [int(x > 0) for x in user_test[test_unseen]]

    # The rest are "validation samples" that we iterate over
    unseen = unseen[num_test:] 

    new_samples = []
    new_labels = []

    for tr in range(trials):

        # Looking for point with smallest minimum distance
        min_seen = [float('inf'), -1]
        for i in [x for x in unseen if (x not in new_samples)]:
            v1 = items[i,:]
            for j in (liked+new_samples):
                v2 = items[j,:]
                dist = euclidean_distance(v1, v2)
                if dist < min_seen[0]:
                    min_seen = [dist, i]

        print(X_titles[min_seen[1]])
        new_samples.append(min_seen[1])
        new_labels.append(map_labels(user_test[min_seen[1]]))
        
        # All items
        plt.subplot(int(np.sqrt(trials)),int(np.sqrt(trials)),tr+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.scatter(items[:,0], items[:,1], alpha=0.1, c=['black']*items.shape[0])
        plt.scatter(items[liked,0], items[liked,1], s=50,c=['green']*len(liked), alpha=0.25)
        plt.scatter(items[disliked,0], items[disliked,1], s=50,c=['red']*len(disliked), alpha=0.25)
        plt.scatter(items[new_samples,0], items[new_samples,1], s=100,c=new_labels, alpha=1.00)
        
        # Plot model
        C = 1.0
        rated = liked+disliked
        X_train = np.concatenate((items[rated,:], items[new_samples,:]))
        y_train = [int(x > 0) for x in np.concatenate((user_train[rated], user_test[new_samples]))]
        clf = svm.SVC(kernel='linear', C=C).fit(X=X_train, y=y_train)
        #clf = svm.SVC(kernel='poly', degree=3, C=C).fit(X=X, y=y)
        #clf = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X=X, y=y)
        #clf = KNeighborsClassifier(n_neighbors=1).fit(X=X_train,y=y_train)
        
        # Append errors
        train_err.append(clf.score(X_train, y_train))
        test_err.append(clf.score(X_test, y_test))
        
        plot_mesh(X_svd, clf)
    plt.suptitle('Greedy Recommendation')
    
    plt.show()

    x = range(trials)
    plt.plot(x, train_err, label='Train')
    plt.plot(x, test_err, label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Additional Samples')
    plt.ylabel('Performance')
    plt.show()


def antigreedy_select(user_train, user_test, items, trials=10, num_test=50):
    '''
    Select points according to whose minimum distance to any previously
    seen data point is maximal

    Distance Metric: Euclidean
    '''

    # Find positively rated samples
    liked = [i for i in user_train.nonzero()[0] if user_train[i] > 0]
    disliked = [i for i in user_train.nonzero()[0] if user_train[i] < 0]
    unseen = user_test.nonzero()[0]

    # Reserve first 50 samples in test as actual test samples
    test_unseen = unseen[:num_test]
    test_err = []
    train_err = []

    # Maintain static test set
    X_test = items[test_unseen,:]
    y_test = [int(x > 0) for x in user_test[test_unseen]]

    # The rest are "validation samples" that we iterate over
    unseen = unseen[num_test:] 

    new_samples = []
    new_labels = []

    for tr in range(trials):

        # Looking for point with largest minimum distance
        max_min_seen = [float('-inf'), -1]

        # For each potential sample
        for i in [x for x in unseen if x not in new_samples]:
            v1 = items[i,:]
            
            # Calculate the minimum distance
            min_seen = [float('inf'), -1]

            # To all previously seen samples
            for j in (liked+new_samples+disliked):
                v2 = items[j,:]
                dist = euclidean_distance(v1, v2)
                if dist < min_seen[0]:
                    min_seen = [dist, i]

            if min_seen[0] > max_min_seen[0]:
                max_min_seen = min_seen

        print(X_titles[max_min_seen[1]])
        new_samples.append(max_min_seen[1])
        new_labels.append(map_labels(user_test[max_min_seen[1]]))
        
        # All items
        plt.subplot(int(np.sqrt(trials)),int(np.sqrt(trials)),tr+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.scatter(items[:,0], items[:,1], alpha=0.1, s=5, c=['black']*items.shape[0])
        plt.scatter(items[unseen,0], items[unseen,1], s=50, c=['black']*items[unseen,:].shape[0], alpha=0.25)
        plt.scatter(items[liked,0], items[liked,1], s=50,c=['green']*len(liked), alpha=0.25)
        plt.scatter(items[disliked,0], items[disliked,1], s=50,c=['red']*len(disliked), alpha=0.25)
        plt.scatter(items[new_samples,0], items[new_samples,1], s=100,c=new_labels, alpha=1.00)
        
        # Plot model
        C = 1.0
        rated = liked+disliked
        X_train = np.concatenate((items[rated,:], items[new_samples,:]))
        y_train = [int(x > 0) for x in np.concatenate((user_train[rated], user_test[new_samples]))]

        #clf = svm.SVC(kernel='linear', C=C).fit(X=X, y=y)
        #clf = svm.SVC(kernel='poly', degree=3, C=C).fit(X=X, y=y)
        #clf = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X=X, y=y)
        clf = KNeighborsClassifier(n_neighbors=1).fit(X=X_train,y=y_train)
        plot_mesh(X_svd, clf)

        # Append errors
        train_err.append(clf.score(X_train, y_train))
        test_err.append(clf.score(X_test, y_test))
    
    plt.suptitle('Antigreedy Recommendation')

    plt.show()

    x = range(trials)
    plt.plot(x, train_err, label='Train')
    plt.plot(x, test_err, label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Additional Samples')
    plt.ylabel('Performance')
    plt.show()

    return

def active_select(user_train, user_test, items, trials=10, num_test=50):
    """
    Select points according to the classifier
    """
    # Find positively rated samples
    liked = [i for i in user_train.nonzero()[0] if user_train[i] > 0]
    disliked = [i for i in user_train.nonzero()[0] if user_train[i] < 0]
    unseen = user_test.nonzero()[0]

    # Reserve first 50 samples in test as actual test samples
    test_unseen = unseen[:num_test]
    test_err = []
    train_err = []

    # Maintain static test set
    X_test = items[test_unseen,:]
    y_test = [int(x > 0) for x in user_test[test_unseen]]

    # The rest are "validation samples" that we iterate over
    unseen = unseen[num_test:] 

    new_samples = []
    new_labels = []

    # Hyperparameters
    C = 1.0
    clf = None

    for tr in range(trials):

        if not clf:
            # Construct model, otherwise use old model
            rated = liked+disliked
            X = np.concatenate((items[rated,:], items[new_samples,:]))
            y = [int(x > 0) for x in np.concatenate((user_train[rated], user_test[new_samples]))]
            #clf = svm.SVC(kernel='linear', C=C, probability=True).fit(X=X, y=y)
            clf = KNeighborsClassifier(n_neighbors=1).fit(X=X,y=y)

        u = [i for i in unseen if i not in new_samples]

        y_pred = clf.predict_proba(items[u,:])

        # Point with class probabilities closest to 0.5
        i = np.argmin(abs(y_pred[:,0] - 0.5))
        sample = u[i]

        print(X_titles[sample])
        new_samples.append(sample)
        new_labels.append(map_labels(user_test[sample]))

        # All items
        plt.subplot(int(np.sqrt(trials)),int(np.sqrt(trials)),tr+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.scatter(items[:,0], items[:,1], alpha=0.1, s=5, c=['black']*items.shape[0])
        plt.scatter(items[unseen,0], items[unseen,1], s=50, c=['black']*items[unseen,:].shape[0], alpha=0.25)
        plt.scatter(items[liked,0], items[liked,1], s=50,c=['green']*len(liked), alpha=0.25)
        plt.scatter(items[disliked,0], items[disliked,1], s=50,c=['red']*len(disliked), alpha=0.25)
        plt.scatter(items[new_samples,0], items[new_samples,1], s=100,c=new_labels, alpha=1.00)
        
        # Plot model
        rated = liked+disliked
        X_train = np.concatenate((items[rated,:], items[new_samples,:]))
        y_train = [int(x > 0) for x in np.concatenate((user_train[rated], user_test[new_samples]))]
        #clf = svm.SVC(kernel='linear', C=C).fit(X=X, y=y)
        #clf = svm.SVC(kernel='poly', degree=3, C=C).fit(X=X, y=y)
        #clf = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X=X, y=y)
        clf = KNeighborsClassifier(n_neighbors=1).fit(X=X_train,y=y_train)
        plot_mesh(X_svd, clf)

        # Append errors
        train_err.append(clf.score(X_train, y_train))
        test_err.append(clf.score(X_test, y_test))
    
    plt.suptitle('Active Recommendation')

    plt.show()

    x = range(trials)
    plt.plot(x, train_err, label='Train')
    plt.plot(x, test_err, label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Additional Samples')
    plt.ylabel('Performance')
    plt.show()

    return

if __name__ == "__main__":
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

    df = pandas.read_csv(data_dir + "u.item", sep="|", header=-1, encoding='latin-1')
    values = df.values

    # Maintain genre data, movie_title
    X_titles = values[:,1]

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
    k = 2 # 50 for TSNE
    train_scores = X_train[(train_rows, train_cols)]
    test_scores = X_test[(test_rows, test_cols)]

    # Now take SVD of X_train
    # (n x k) . (k x k) . (k x d)
    #U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
    # What we really want are movie vectors (d x k)
    #X_svd = np.transpose(np.diag(s[0:k]).dot(Vt[0:k, :]))
    #np.save('ml_100k_svd.npy', X_svd)

    X_svd = np.load('ml_100k_svd.npy')

    #model = TSNE(n_components=2, random_state=0)
    #X_svd = model.fit_transform(X_svd) 

    # Peek at explained variance
    #plt.plot(range(s.shape[0]), np.cumsum(s) / np.sum(s))
    #plt.show()

    # Peek at movie representations (2-d only)
    #plt.scatter(X_svd[:,0], X_svd[:,1])
    #plt.show()

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

    # Pick USERS with few train samples for the demo
    #USERS = [i for i,c in train_counts if c < 20]

    # Pick USERS with many train samples for the demo
    USERS = [i for i,c in train_counts if c > 100]
    print(len(USERS))

    # Swap train and test to get more test samples
    X_train, X_test = X_test, X_train

    alg = sys.argv[1]

    for USER in USERS:
    # Note that the furthest point in train may not be the furthest point overall
    # We only have ten test samples per user

        # Set number of trials equal to total number of unseen movies
        trials = 36
        #trials = len(X_test[USER,:].nonzero()[0])

        print("\nSelecting samples for user:", USER)
        if alg == 'greedy':
            greedy_select(X_train[USER,:], X_test[USER,:], X_svd, trials=trials)
        elif alg == 'antigreedy':
            antigreedy_select(X_train[USER,:], X_test[USER,:], X_svd, trials=trials)
        elif alg == 'active':
            active_select(X_train[USER,:], X_test[USER,:], X_svd, trials=trials)
        else:
            print("Input \'greedy\' or \'antigreedy\' or \'active\'")



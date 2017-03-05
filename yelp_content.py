import os
import json
import numpy as np
import pandas as pd

# Do dimensionality reduction here
def dim_reduce(M, k):

    return None

# Load data in here and convert to matrix
# (Load more data later)
def load_data(business_file):
    max_data = 10
    business_datalist = []
    with open(business_file) as business_data:
        for line in business_data:
            if max_data:
                business_datalist.append(json.loads(line))
                max_data -=1
            else:
                break

    columns = []
    for bus in business_datalist:
        # sort values in alphabetical order of their keys
        sorted_kvs = sorted([(k, v) for (k, v) in bus.items()], key=lambda (k, v): k)
        col = [v for (k, v) in sorted_kvs]
        print col
        columns.append(np.array(col))
    business_matrix = np.array(columns)
    return business_matrix.T


def main():
    business_datalist = load_data('../yelp_data/yelp_academic_dataset_business.json');

if __name__ == "__main__":
    main()
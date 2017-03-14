import os
import json
import numpy as np
import pandas as pd
from yelp_featurize import Business

# Do dimensionality reduction here
def dim_reduce(M, k):
    # we only have 5 features lol
    return None


# Load data in here and convert to matrix
# (Load more data later)
def load_data(business_file):
    max_data = 10
    business_datalist = []
    with open(business_file) as business_data:
        for line in business_data:
            business_datalist.append(json.loads(line))

    columns = []
    print_ct = 1
    for bus in business_datalist:
        b = Business(bus)
        col = b.featurize()
        columns.append(col)
    business_matrix = np.array(columns)
    print (business_matrix.T).shape
    return business_matrix.T


def main():
    business_datalist = load_data('../yelp_data/yelp_academic_dataset_business.json');

if __name__ == "__main__":
    main()
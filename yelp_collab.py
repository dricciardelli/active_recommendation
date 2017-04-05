import os
import json
import numpy as np
import pandas as pd
from yelp_featurize import Business

# Partially load ratings matrix here
def load_data(ratings_file):
    reviews_data = pd.read_csv(ratings_file,
                                 nrows = 100)
    print(reviews_data.iloc[1000, :].text)
    return review_data

def main():
    business_datalist = load_data('./data/yelp/yelp_academic_dataset_review.json');

if __name__ == "__main__":
    main()
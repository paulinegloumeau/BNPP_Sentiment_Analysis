import os
import errno
import pandas as pd

import json

raw_data_path = './data_yelp/raw/'
preprocessed_data_path = './data_yelp/preprocessed/'

def create_file(filename):
    # Also works for creating a directory
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def split_raw_json_review_file(lines_per_file = 100000, max_files = 1):
    files_created = 0
    smallfile = None
    with open(raw_data_path + 'json/yelp_academic_dataset_review.json') as big_file:
        for lineno, line in enumerate(big_file):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = preprocessed_data_path + 'yelp_academic_dataset_review_split/yelp_academic_dataset_review_{}.json'.format(lineno + lines_per_file)
                create_file(small_filename)
                smallfile = open(small_filename, "w")
                files_created += 1
            smallfile.write(line)
            if files_created == max_files + 1:
                break
        if smallfile:
            smallfile.close()  

def split_raw_csv_review_file(lines_per_file = 100000, single_file = True, join_business = True):
    create_file(preprocessed_data_path + 'yelp_academic_dataset_review_split/')
    if join_business:
        business_df = pd.read_csv(raw_data_path + 'csv/yelp_business.csv')
    for i,chunk in enumerate(pd.read_csv(raw_data_path + 'csv/yelp_review.csv', chunksize=lines_per_file)):
        if join_business:
            chunk = pd.merge(left=chunk, right=business_df, on='business_id', how='left')
            chunk = chunk[['review_id', 'business_id', 'stars_x', 'stars_y', 'date', 'city', 'text', 'categories']]
        chunk.to_csv(preprocessed_data_path + 'yelp_academic_dataset_review_split/yelp_academic_dataset_review_{}.csv'.format((i+1)*lines_per_file), index='False')
        if single_file:
            break

split_raw_csv_review_file()

# def join_review_business_data(review_file = './data_yelp/yelp_academic_dataset_review_json_split/yelp_academic_dataset_review_100000.json'):
#     review_df = pd.read_json(review_file, lines=True)[['text']]
#     business_df = pd.read_json('./data_yelp/yelp_academic_dataset_business.json', lines=True)[['text']]
#     print(business_df.head())
#     final_df = pd.merge(left=review_df, right=business_df, on='business_id', how='left')
#     print(final_df.head())




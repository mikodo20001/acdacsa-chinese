import random
import re

import pandas as pd
import os
from tqdm import tqdm

random.seed(0)

# cate_columns = ['Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find', 'Service#Queue',
#                 'Service#Hospitality', 'Service#Parking', 'Service#Timely', 'Price#Level', 'Price#Cost_effective',
#                 'Price#Discount', 'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
#                 'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend']
cate_columns = ['Location#Easy_to_find', 'Service#Timely', 'Price#Level', 'Ambience#Space', 'Food#Portion']
# cate_sample_num = 5
discard_prob = 0
file_last = ""
data_dir = "./asap"
output_dir = "./asap_data" + file_last
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_file_path = os.path.join(data_dir, "train" + file_last + ".csv")
valid_file_path = os.path.join(data_dir, "dev" + file_last + ".csv")
test_file_path = os.path.join(data_dir, "test" + file_last + ".csv")

train_out_file_path = os.path.join(output_dir, "train.txt")
valid_out_file_path = os.path.join(output_dir, "val.txt")
test_out_file_path = os.path.join(output_dir, "test.txt")

train_data = pd.read_csv(train_file_path, encoding="utf-8")
valid_data = pd.read_csv(valid_file_path, encoding="utf-8")
test_data = pd.read_csv(test_file_path, encoding="utf-8")


def generate_data(train_data, output_file_path, is_train=True):
    with open(output_file_path, "w+", encoding="utf-8") as f:
        pbar = tqdm(total=train_data.shape[0])
        for idx, row in train_data.iterrows():
            pbar.update(1)

            review = row['review']
            for cate_column in cate_columns:
                if is_train and random.random() < discard_prob:
                    continue
                cate = row[cate_column]
                if cate not in [1, 0, -1]:
                    continue
                cate_desc = "neutral"
                if cate > 0:
                    cate_desc = "positive"
                if cate < 0:
                    cate_desc = "negative"

                cate_ = " ".join(re.split(r"#|_", cate_column)).lower()
                line = f"{review}\001The sentiment polarity of {cate_} is {cate_desc} ."
                if not is_train:
                    line = f"{review}\001{cate_}\001{cate_desc}"
                f.write(line + "\n")


generate_data(train_data, train_out_file_path)
generate_data(valid_data, valid_out_file_path, is_train=False)
generate_data(test_data, test_out_file_path, is_train=False)

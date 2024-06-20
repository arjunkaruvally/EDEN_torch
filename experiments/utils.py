import os
import sys
import pandas as pd
from tqdm import tqdm
import json


def aggregate_results(path):
    json_list = []
    for file_id, file in enumerate(os.listdir(path)):
        if file.endswith(".json"):
            jsonpath = os.path.join(path, file)
            if file_id % 10 == 0:
                print(f"Processing {jsonpath} {file_id}/{len(os.listdir(path))}")
            with open(jsonpath, 'r') as f:
                json_list.append(json.load(f))

    df = pd.DataFrame(json_list)
    return df


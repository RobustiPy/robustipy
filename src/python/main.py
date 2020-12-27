import os
from load_data import load_union_data

if __name__ == "__main__":
    d_path = os.path.join(os.getcwd(), '..','..', 'data')
    union_df = load_union_data()



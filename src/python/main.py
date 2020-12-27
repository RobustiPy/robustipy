import os
from load_data import load_union_data


def run_union_example(d_path):
    union_df = load_union_data(d_path)
    print(union_df)


def main():
    d_path = os.path.join(os.getcwd(), '..', '..', 'data')
    run_union_example(d_path)
    # @TODO: new replication one
    # @TODO: new replication two
    # @TODO: new replication three


if __name__ == "__main__":
    main()

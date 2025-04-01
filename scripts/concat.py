import pandas as pd
from tqdm import trange

def main():
    dfs = (pd.read_csv(f"./data/data_{i}.csv") for i in trange(0, 151))
    pd.concat(dfs).to_csv('final_data.csv', index=False)


if __name__ == '__main__':
    main()

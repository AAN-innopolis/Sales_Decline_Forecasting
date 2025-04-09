import os
import sys
import pandas as pd
from sodapy import Socrata
from tqdm import tqdm
import time
from random import randint

BATCH_SIZE = 10 ** 5
DATA_SET_SIZE = 32_000_000

SLEEP_DYNAMIC_TIME = 20


def save_data_frame(offset=0):
    client = Socrata("data.iowa.gov", None)
    results = client.get("m3tr-qhgy", order="date DESC", offset=offset, limit=BATCH_SIZE)
    results_df = pd.DataFrame.from_records(results)
    results_df.to_csv(f"data_{offset // BATCH_SIZE}.csv", index=False)


def try_save(offsetId=0):
    global SLEEP_DYNAMIC_TIME
    try:
        save_data_frame(offsetId * BATCH_SIZE)
        SLEEP_DYNAMIC_TIME = 20
        sleep_time = randint(10, 20)
        time.sleep(sleep_time)
    except:

        SLEEP_DYNAMIC_TIME *= 2
        print("Failed to get data id=" + str(offsetId) + " Retrying... in " + str(SLEEP_DYNAMIC_TIME))
        time.sleep(SLEEP_DYNAMIC_TIME)
        try_save(offsetId)


def main():
    skip_until = 0
    if len(sys.argv) == 2:
        skip_until = int(sys.argv[1])

    for i in tqdm(range(DATA_SET_SIZE // BATCH_SIZE)):
        if i < skip_until:
            continue
        if os.path.isfile(f"data_{i}.csv"):
            print(f"Assuming data batch id={i} already downloaded. Skipping...")
            continue
        try_save(i)
        sleep_time = randint(2, 20)
        time.sleep(sleep_time)


if __name__ == '__main__':
    main()

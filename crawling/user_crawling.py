import json
import requests
import pandas as pd
import argparse
import logging
import time
import os


def search_users(page):
    url = f"https://solved.ac/api/v3/ranking/tier?page={page}"
    users = requests.get(url)
    if users.status_code == requests.codes.ok:
        users = json.loads(users.content.decode("utf-8"))
        users = users.get("items")
        df = pd.DataFrame(data=users)
        return df
    else:
        print(f"{users.status_code} Error")


def main(args):
    start, end = args.start, args.end
    for page_num in range(start, end + 1):
        user_df = search_users(page_num)
        if start == page_num:
            users_df = user_df
        else:
            users_df = pd.concat([users_df, user_df], ignore_index=True)

        print(f"{page_num} page Done")

        time.sleep(3.5156)

    users_df.to_csv(os.path.join(args.save_path, f"{start}_{end}_user.csv"))


if __name__ == "__main__":
    # Setup argparse to get settings
    parser = argparse.ArgumentParser(description="Set settings for crawling problems.")
    parser.add_argument(
        "-l", "--logging", type=str, default="DEBUG", help="Set logging level."
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=1001,
        help="The start of the user ranking page number.",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=1005,
        help="The end of the user ranking page number.",
    )
    parser.add_argument(
        "--errornum",
        type=int,
        default=5,
        help="Number of times the code can get a unknown response code before exiting.",
    )
    parser.add_argument(
        "--save_path", type=str, default="./data", help="save data path information"
    )

    # Get arguments
    args = parser.parse_args()

    data_path = args.save_path

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    main(args=args)

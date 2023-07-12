import argparse
import datetime
import logging
import math
import os
import pandas as pd
import requests
import signal
import sys
import time


def main(args, logger) -> None:
    """
    Checks through the problem numbers to get the data and save it as a csv file

    Parameters:
        args(dict) Contains the settings used
            max_error_num(int) Max number of error before finishing program
        logger(logging.Logger) Logs process through the process
    """

    # Create saved data folder if it doesn't exist
    if not os.path.isdir("data"):
        logger.info("Data file missing...\nCreating data file")
        os.mkdir("data")
        logger.debug("Data file created")

    # Used to save the results
    output_df = pd.DataFrame()

    # URL used to get data using API
    url = "https://solved.ac/api/v3/search/problem"
    headers = {"Accept": "application/json"}

    logger.debug("Starting crawling")

    # Checks for number of consecutive errors
    error_num = 0

    # Set current page and total page
    total_page_num = None
    page_num = 1

    # While problem number is not over end problem number
    while total_page_num is None or page_num <= total_page_num:
        # Set problem to get question number
        querystring = {
            "query": "",
            "page": page_num,
        }

        # Get response from HTTP request
        response = requests.get(url, headers=headers, params=querystring)

        logger.debug(f"Page Number: {page_num}\tStatus Code: {response.status_code}")

        # If response code from HTTP request is OK
        if response.status_code == 200:
            # Get max page number if the number is not determined
            if total_page_num is None:
                total_page_num = math.ceil(response.json()["count"] / 50)

            # Update dataframe using data
            output_df = pd.concat(
                [output_df, pd.DataFrame(response.json()["items"])], ignore_index=True
            )

            logger.info(f"Problem Num {page_num} Done!!!")

            # Update page number
            page_num += 1

            # Reset consecutive error count
            error_num = 0

        # If response code from HTTP request is not any of the above
        else:
            logger.info(f"Problem Num {page_num} Unknown Response Code")

            # Update consecutive error count
            error_num += 1

            # If the number of consecutive errors are equal to or over the settings
            if error_num >= args.max_error_num:
                # Save results
                output_df.to_csv(
                    os.path.join("data", f"problem_{page_num}_data.csv"), index=False
                )

                logger.warning(f"{error_num} errors reached. Ending program")

                return

        # Wait enough time to get result
        # 256 queries for 15 minutes -> 1 query for 3.515625
        time.sleep(3.5156)

    # Remove duplicates users
    output_df.drop_duplicates("handle", ignore_index=True)

    # Save output dataframe as csv file
    output_df.to_csv(os.path.join("data", f"problem_data.csv"))

    logger.debug("Saved CSV file")

    return


if __name__ == "__main__":
    # Setup argparse to get settings
    parser = argparse.ArgumentParser(description="Set settings for crawling problems.")
    parser.add_argument(
        "-l", "--logging", type=str, default="DEBUG", help="Set logging level."
    )
    parser.add_argument(
        "-e",
        "--max_error_num",
        type=int,
        default=5,
        help="Number of times the code can get a unknown response code before exiting.",
    )

    # Get arguments
    args = parser.parse_args()

    # Setup logger
    # Create log folder
    if not os.path.isdir("log"):
        os.mkdir("log")

    # Get starting date
    create_time = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    # Create log file
    f = open(f"{os.path.join('log', create_time)}.txt", "w")

    # Set basic log settings
    logging.basicConfig(
        filename=os.path.join("log", f"{create_time}.txt"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    # Set logging level
    logger = logging.getLogger()
    logger.setLevel(args.logging)

    # Set output logging
    logger.addHandler(logging.StreamHandler())

    logger.debug("Starting Program")

    main(args, logger)

    logger.debug("Ending Program")

import argparse
import logging
import math
import os
import pandas as pd
import requests
import time


def main(args, logger) -> None:
    """
    Checks through each user to get the solved/unsolved problem data and save it as a csv file

    Parameters:
        args(dict) Contains the settings used
            start_user_index(int) Starting user index number
            end_user_index(int) Ending user index number
            max_error_num(int) Max number of error before finishing program
        logger(logging.Logger) Logs process through the process
    """

    # Create saved data folder if it doesn't exist
    if not os.path.isdir("data"):
        logger.info("Data file missing...\nCreating data file")
        os.mkdir("data")
        logger.debug("Data file created")

    # Used to save the results as csv
    correct_output_df = pd.DataFrame()
    tried_output_df = pd.DataFrame()

    # Check if starting user index is smaller than the ending user index
    if args.start_user_index > args.end_user_index:
        logger.error("Starting user index number higher than ending user index number!")
        return

    # Check if user data is in data folder
    if not os.path.exists(os.path.join("data", "user_data.csv")):
        logger.warning("User data file not found")

        return

    # Get user ID list from user data and starting / ending index
    user_list = pd.read_csv(os.path.join("data", "user_data.csv"))["handle"][
        args.start_user_index : args.end_user_index + 1
    ].to_list()

    # URL used to get data using API
    url = "https://solved.ac/api/v3/search/problem"
    headers = {"Accept": "application/json"}

    logger.debug("Starting crawling")

    # Checks for error numbers in code until termination
    error_num = 0

    # Loop through every user ID to get interaction data
    for temp_index, temp_user in enumerate(user_list):
        logging.info(
            f"Index number: {temp_index + args.start_user_index}\tUser name: {temp_user}"
        )

        # Get correct problem data from user ID
        # Set up total page and current page number
        total_page_num = None
        page_num = 1

        # While start problem page number is not over end problem page number
        while total_page_num is None or total_page_num >= page_num:
            # Set query to get problem page number
            querystring = {
                "query": f"s@{temp_user}",
                "page": page_num,
            }

            # Get response from HTTP request
            response = requests.get(url, headers=headers, params=querystring)

            logger.debug(f"User Name: {temp_user}\tStatus Code: {response.status_code}")

            # If response code from HTTP request is OK
            if response.status_code == 200:
                # Get max page number if the number is not determined
                if total_page_num is None:
                    total_page_num = math.ceil(response.json()["count"] / 50)

                # Get correct problems as dataframe
                temp_df = pd.DataFrame(response.json()["items"])
                temp_df["userId"] = temp_user
                temp_df = temp_df[["userId", "problemId"]]

                # Update dataframe using data
                correct_output_df = pd.concat(
                    [correct_output_df, temp_df], ignore_index=True
                )

                logger.info(f"Page Num {page_num} Done!!!")

                # Reset error count
                error_num = 0

                # Update page number
                page_num += 1

            # If response code from HTTP request is not any of the above
            else:
                logger.info(f"Page Num {page_num} Unknown Response Code")

                # Update error count
                error_num += 1

                # If the number of errors are over the settings
                if error_num >= args.errornum:
                    # Save current results
                    correct_output_df.to_csv(
                        os.path.join(
                            "data",
                            f"correct_{args.start_user_index}_{args.start_user_index + temp_index}_data.csv",
                        ),
                        index=False,
                    )

                    logger.warning(f"{error_num} errors reached. Ending program")

                    return

            # Wait enough time to get result
            # 256 queries for 15 minutes -> 1 query for 3.515625
            time.sleep(3.5156)

        # Remove duplicates questions
        correct_output_df.drop_duplicates(["userId", "problemId"], ignore_index=True)

        # Get tried problem data from user ID
        # Set up total page and current page number
        total_page_num = None
        page_num = 1

        while total_page_num is None or total_page_num >= page_num:
            # Set query to get problem page number
            querystring = {
                "query": f"t@{temp_user}",
                "page": page_num,
            }

            # Get response from HTTP request
            response = requests.get(url, headers=headers, params=querystring)

            logger.debug(f"User Name: {temp_user}\tStatus Code: {response.status_code}")

            # If response code from HTTP request is OK
            if response.status_code == 200:
                # Get max page number if the number is not determined
                if total_page_num is None:
                    total_page_num = math.ceil(response.json()["count"] / 50)

                # Get tried problems as dataframe
                temp_df = pd.DataFrame(response.json()["items"])
                temp_df["userId"] = temp_user
                temp_df = temp_df[["userId", "problemId"]]

                # Update dataframe using data
                tried_output_df = pd.concat(
                    [tried_output_df, temp_df], ignore_index=True
                )

                logger.info(f"Page Num {page_num} Done!!!")

                # Reset error count
                error_num = 0

                page_num += 1

            # If response code from HTTP request is not any of the above
            else:
                logger.info(f"Page Num {page_num} Unknown Response Code")

                error_num += 1

                # If the number of errors are over the settings
                if error_num >= args.errornum:
                    # Save current results
                    correct_output_df.to_csv(
                        os.path.join(
                            "data",
                            f"correct_{args.start_user_index}_{args.start_user_index + temp_index}_data.csv",
                        ),
                        index=False,
                    )
                    tried_output_df.to_csv(
                        os.path.join(
                            "data",
                            f"tried_{args.start_user_index}_{args.start_user_index + temp_index}_data.csv",
                        ),
                        index=False,
                    )

                    logger.warning(f"{error_num} errors reached. Ending program")

                    return

            # Wait enough time to get result
            # 256 queries for 15 minutes -> 1 query for 3.515625
            time.sleep(3.5156)

        # Remove duplicates questions
        tried_output_df.drop_duplicates(["userId", "problemId"], ignore_index=True)

        correct_output_df.to_csv(
            os.path.join("data", f"temp_correct_data.csv"), index=False
        )
        tried_output_df.to_csv(
            os.path.join("data", f"temp_tried_data.csv"), index=False
        )

    # Get incorrect output dataframe
    correct_output_df["answerCode"] = 1

    # Merage data
    tried_output_df = tried_output_df.merge(
        correct_output_df, on=["userId", "problemId"], how="left"
    )

    # All other N/A columns are problems that the user got incorrect
    tried_output_df = tried_output_df.fillna(0)

    # Save output dataframe as csv file
    tried_output_df.to_csv(
        os.path.join(
            "data",
            f"interaction_{args.start_user_index}_{args.end_user_index}_data.csv",
        ),
        index=False,
    )

    logger.info("Saved CSV file")

    # Remove temp files from data file
    os.remove(os.path.join("data", f"temp_correct_data.csv"))
    os.remove(os.path.join("data", f"temp_tried_data.csv"))

    return


if __name__ == "__main__":
    # Setup argparse to get settings
    parser = argparse.ArgumentParser(description="Set settings for crawling problems.")
    parser.add_argument(
        "-l", "--logging", type=str, default="INFO", help="Set logging level."
    )
    parser.add_argument(
        "-s",
        "--start_user_index",
        type=int,
        default=10000,
        help="The start of the problem page number.",
    )
    parser.add_argument(
        "-e",
        "--end_user_index",
        type=int,
        default=10001,
        help="The end of the problem page number.",
    )
    parser.add_argument(
        "-m",
        "--max_error_num",
        type=int,
        default=5,
        help="Number of times the code can get a unknown response code before exiting.",
    )

    # Get arguments
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(args.logging)

    logger.debug("Starting Program")

    main(args, logger)

    logger.debug("Ending Program")

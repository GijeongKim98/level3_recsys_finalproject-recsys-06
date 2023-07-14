import argparse
import datetime
import logging
import math
import os
import pandas as pd
import requests
import time


def main(logging):
    """
    Checks through the user data to get the changed users to update the interaction data

    Parameters:
        args(dict) Contains the settings used
            max_error_num(int) Max number of error before finishing program
        logger(logging.Logger) Logs process through the process
    """

    while True:
        start_time = datetime.datetime.now()
        logging.info(f"Start Time: {start_time}")

        # Create temporary 
        comp_user_df = pd.DataFrame()

        # Setup current page and the maximum page number
        page_num = 1
        total_page_num = None

        while total_page_num is None or total_page_num >= page_num:
            # Set query to get user data from page number
            url = f"https://solved.ac/api/v3/ranking/tier?page={page_num}"

            logging.debug(f"Getting User Data Page: {page_num}")

            # Get the maximum page number
            if total_page_num is None:
                total_page_num = math.ceil(response.json()["count"] / 50)

            # Get response from HTTP request
            response = requests.get(url)

            time.sleep(3.5156)

            # If the status is OK
            if response.status_code == 200:
                # Get user data
                comp_user_df = pd.concat(
                    [comp_user_df, pd.DataFrame(response.json()["items"])],
                    ignore_index=True,
                )

                # Update current page number
                page_num += 1

            # If response code from HTTP request is not any of the above
            else:
                continue

        # Combine both dataframes
        temp_user_df = comp_user_df[["handle", "solvedCount"]].rename(
            columns={"solvedCount": "compSolvedCount"}
        )
        input_user_df = pd.read_csv(os.path.join("data", "user_data.csv"))
        temp_user_df = temp_user_df.merge(
            input_user_df[["handle", "solvedCount"]], on="handle"
        )

        # Get the list of users with different solved problem numbers
        changed_user = temp_user_df[
            (temp_user_df["solvedCount"] * 1.01).apply(lambda x: math.floor(x))
            <= temp_user_df["compSolvedCount"]
        ]["handle"].unique()

        # Update user data with the ID used above
        input_user_df = input_user_df.drop(
            input_user_df.loc[input_user_df["handle"].isin(changed_user)].index
        )
        input_user_df = pd.concat(
            [input_user_df, comp_user_df.loc[comp_user_df["handle"].isin(changed_user)]]
        )

        # Drop the duplicates with the same ID and keep the data with the most solved problem numbers
        input_user_df = (
            input_user_df.sort_values("solvedCount", ascending=False)
            .drop_duplicates("handle")
            .sort_index(ignore_index=True)
        )

        # Save user data
        input_user_df.to_csv(os.path.join("data", "user_data.csv"))

        logging.info(f"Number of Users Updating: {len(changed_user)}")

        if len(changed_user) == 0:
            logging.info("No updates needed, restarting user crawling")

            continue

        # URL used to get interaction data using API
        url = "https://solved.ac/api/v3/search/problem"
        headers = {"Accept": "application/json"}

        # Get interaction data to get updates
        input_interact_df = pd.read_csv(os.path.join("data", "interaction_data.csv"))

        # Used to store correct / tried problems
        correct_output_df = pd.DataFrame()
        tried_output_df = pd.DataFrame()

        # Loop through the changed user list to get interactions
        for temp_index, temp_user in enumerate(changed_user):
            logging.debug(
                f"Getting Problem Data from User: {temp_user} ({temp_index + 1}/{len(changed_user)})"
            )

            # Drop all user interactions
            input_interact_df = input_interact_df.drop(
                input_interact_df.loc[input_interact_df["userId"] == temp_user].index
            )

            logging.debug("Getting Correct Problems:")

            # Setup current page and the maximum page number
            total_page_num = None
            temp_page_num = 1

            while total_page_num is None or total_page_num >= temp_page_num:
                logging.debug(f"Running Correct Page {temp_page_num}")

                # Set query to get correct problems using user and page number
                querystring = {
                    "query": f"s@{temp_user}",
                    "page": temp_page_num,
                }

                # Get response from HTTP request
                response = requests.get(url, headers=headers, params=querystring)

                time.sleep(3.5156)

                # If response code from HTTP request is OK
                if response.status_code == 200:
                    # Get maximum page number
                    if total_page_num is None:
                        total_page_num = math.ceil(response.json()["count"] / 50)

                    # Get correct dataframe
                    temp_df = pd.DataFrame(response.json()["items"])
                    temp_df["userId"] = temp_user
                    temp_df = temp_df[["userId", "problemId"]]

                    # Update dataframe using data
                    correct_output_df = pd.concat(
                        [correct_output_df, temp_df], ignore_index=True
                    )

                    # Update page number
                    temp_page_num += 1

                # If response code from HTTP request is not any of the above
                else:
                    continue

            total_page_num = None
            temp_page_num = 1

            print("Getting Tried Problems:")

            while total_page_num is None or total_page_num >= temp_page_num:
                print(f"Running Tried Page {temp_page_num}")

                # Set query to get tried problems using user and page number
                querystring = {
                    "query": f"t@{temp_user}",
                    "page": temp_page_num,
                }

                # Get response from HTTP request
                response = requests.get(url, headers=headers, params=querystring)

                time.sleep(3.5156)

                # If response code from HTTP request is OK
                if response.status_code == 200:
                    # Get total page number
                    if total_page_num is None:
                        total_page_num = math.ceil(response.json()["count"] / 50)

                    # Get tried dataframe
                    temp_df = pd.DataFrame(response.json()["items"])
                    temp_df["userId"] = temp_user
                    temp_df = temp_df[["userId", "problemId"]]

                    # Update dataframe using data
                    tried_output_df = pd.concat(
                        [tried_output_df, temp_df], ignore_index=True
                    )

                    # Update page number
                    temp_page_num += 1

                # If response code from HTTP request is not any of the above
                else:
                    continue

            # Temporarily save correct and tried data
            correct_output_df.to_csv(
                os.path.join("data", f"temp_correct_data.csv"), index=False
            )
            tried_output_df.to_csv(
                os.path.join("data", f"temp_tried_data.csv"), index=False
            )

        # Merge both dataframes to get interaction data
        correct_output_df["answerCode"] = 1
        tried_output_df = tried_output_df.merge(
            correct_output_df, on=["userId", "problemId"], how="left"
        )
        tried_output_df = tried_output_df.fillna(0)

        # update data to interaction data
        tried_output_df = pd.concat(
            [input_interact_df, tried_output_df], ignore_index=True
        )

        # Save output data
        tried_output_df.to_csv(
            os.path.join("data", f"interaction_data.csv"), index=False
        )

        # Remove temporary data
        os.remove(os.path.join("data", f"temp_correct_data.csv"))
        os.remove(os.path.join("data", f"temp_tried_data.csv"))

        end_time = datetime.datetime.now()
        logging.info(f"End Time: {end_time}")

        logging.debug(f"End Time: {end_time}")
        logging.info(f"Total Time Used: {end_time - start_time}")


if __name__ == "__main__":
    # Setup argparse to get settings
    parser = argparse.ArgumentParser(description="Set settings for crawling problems.")
    parser.add_argument(
        "-l", "--logging", type=str, default="DEBUG", help="Set logging level."
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
    f = open(os.path.join("log", f"server_{create_time}.txt"), "w")

    # Set basic log settings
    logging.basicConfig(
        filename=os.path.join("log", f"server_{create_time}.txt"),
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

    logger.debug("Starting Server")

    main(logger)

    logger.debug("Ending Server")
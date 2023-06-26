import argparse
import logging
import os
import pandas as pd
import requests
import signal
import sys
import time


# Used to setup behavior after ctrl+c. Used to save the current dataframe
def SignalHandler_SIGINT(SignalNumber,Frame):
    global output_df

    # Save interrupted file as interrupt_temp.csv
    output_df.to_csv(os.path.join('data', f'interrupt_temp.csv'))

    sys.exit(0)


def main(args, logger) -> None:
    '''
    Checks through the problem numbers to get the data and save it as a csv file

    Parameters:
        args(dict) Contains the settings used
            start(int) Starting problem number
            end(int) Ending problem number
        logger(logging.Logger) Logs process through the process
    '''

    # Create saved data folder if it doesn't exist
    if not os.path.isdir('data'):
        logger.info('Data file missing...\nCreating data file')
        os.mkdir('data')
        logger.debug('Data file created')

    # Used to save the results as csv
    global output_df

    # Get start problem number
    start_prob_num = args.start
    # Get end problem number
    end_prob_num = args.end

    # Check if starting number is smaller than the ending number
    if start_prob_num > end_prob_num:
        logger.error('Starting problem number higher than ending problem number!')
        return

    # URL used to get data using API
    url = "https://solved.ac/api/v3/problem/show"
    headers = {"Accept": "application/json"}

    # Set current problem number
    prob_num = start_prob_num

    logger.debug('Starting crawling')

    # Checks for error numbers in code until termination
    error_num = 0

    # While problem number is not over end problem number
    while prob_num <= end_prob_num:
        # Set problem to get question number
        querystring = {"problemId": prob_num}

        # Get response from HTTP request
        response = requests.get(url, headers=headers, params=querystring)

        logger.debug(f'Problem Number: {prob_num}\tStatus Code: {response.status_code}')

        # If response code from HTTP request is OK
        if response.status_code == 200:
            # Update dataframe using data
            output_df = pd.concat([output_df, pd.DataFrame([response.json()])], ignore_index=True)

            # Add 1 to problem count
            prob_num += 1

            logger.info(f'Problem Num {prob_num} Done!!!')

            # Reset error count
            error_num = 0

        # If response code from HTTP request is NOT FOUND
        elif response.status_code == 404:
            # Add 1 to problem count
            prob_num += 1

            logger.info(f'Problem Num {prob_num} Page Missing')

            # Reset error count
            error_num = 0

        # If response code from HTTP request is not any of the above
        else:
            logger.info(f'Problem Num {prob_num} Unknown Response Code')

            error_num += 1

            # If the number of errors are over the settings
            if error_num >= args.errornum:
                # Save results
                output_df.to_csv(os.path.join('data', f'problem_{prob_num}_{end_prob_num}_data.csv'))
                
                logger.warning(f'{error_num} errors reached. Ending program')

                return

        # Wait enough time to get result
        # 256 queries for 15 minutes -> 1 query for 3.515625
        time.sleep(3.5156)

    # Save output dataframe as csv file
    output_df.to_csv(os.path.join('data', f'problem_{start_prob_num}_{end_prob_num}_data.csv'))

    logger.debug('Saved CSV file')

    return


if __name__ == '__main__':
    # Set output_df as global variable to save after interrupt
    global output_df

    # Set dataframe
    output_df = pd.DataFrame()

    # Setup interrupt actions
    signal.signal(signal.SIGINT,SignalHandler_SIGINT)

    # Setup argparse to get settings
    parser = argparse.ArgumentParser(description='Set settings for crawling problems.')
    parser.add_argument('-l', '--logging', type=str, default='DEBUG', help='Set logging level.')
    parser.add_argument('-s', '--start', type=int, default=1001, help='The start of the problem number.')
    parser.add_argument('-e', '--end', type=int, default=1005, help='The end of the problem number.')
    parser.add_argument('--errornum', type=int, default=5, help='Number of times the code can get a unknown response code before exiting.')

    # Get arguments
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(args.logging)

    logger.debug('Starting Program')

    main(args, logger)

    logger.debug('Ending Program')

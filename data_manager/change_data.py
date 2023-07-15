import argparse
import datetime
import logging
import os
import pandas as pd


def process_user_data(input_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes raw user data to data to be used for training

    Parameters:
        input_df(pd.DataFrame) Raw user data
    Returns:
        changed_input_df(pd.DataFrame) Processed user data
    '''

    # Process data
    changed_input_df = input_df

    return changed_input_df


def process_item_data(input_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes raw item data to data to be used for training

    Parameters:
        input_df(pd.DataFrame) Raw item data
    Returns:
        changed_input_df(pd.DataFrame) Processed item data
    '''

    # Process data
    changed_input_df = input_df

    return changed_input_df


def process_inter_data(input_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes raw interaction data to data to be used for training

    Parameters:
        input_df(pd.DataFrame) Raw interaction data
    Returns:
        changed_input_df(pd.DataFrame) Processed interaction data
    '''

    # Process data
    changed_input_df = input_df

    return changed_input_df


def process_data(setting, logging):
    '''
    Changes raw data to data to be used for training

    Parameters:
        setting(dict) Contains the settings used
            folder_data_file(str) Folder name for data
            user_data_file(str) File name of user data
            item_data_file(str) File name of item data
            inter_data_file(str) File name of interaction data
        logger(logging.Logger) Used for logging
    '''

    # Get folder that contains the raw data
    raw_data_folder = setting['raw_data_folder_path']

    # Get path to data
    user_data_file = os.path.join(raw_data_folder, setting['user_data_file'])
    item_data_file = os.path.join(raw_data_folder, setting['item_data_file'])
    inter_data_file = os.path.join(raw_data_folder, setting['inter_data_file'])

    logging.debug('Getting raw data from path')

    # Get data
    user_df = pd.read_csv(user_data_file)
    item_df = pd.read_csv(item_data_file)
    inter_df = pd.read_csv(inter_data_file)

    logging.debug('Processing user data')

    # Process each data seperatly
    # Process user data
    processed_user_df = process_user_data(user_df)

    logging.debug('Processing item data')

    # Process item data
    processed_item_df = process_item_data(item_df)

    logging.debug('Processing interaction data')

    # Process interaction data
    processed_inter_df = process_inter_data(inter_df)

    logging.debug('Processing extra data')

    # Process data together

    logging.debug('Saving processed data')

    # Save processed data
    processed_data_folder = setting['processed_data_folder_path']

    # Create processed data folder
    if not os.path.isdir(processed_data_folder):
        os.mkdir(processed_data_folder)

    user_data_file = os.path.join(processed_data_folder, setting['user_data_file'])
    item_data_file = os.path.join(processed_data_folder, setting['item_data_file'])
    inter_data_file = os.path.join(processed_data_folder, setting['inter_data_file'])

    processed_user_df.to_csv(user_data_file, index=False)
    processed_item_df.to_csv(item_data_file, index=False)
    processed_inter_df.to_csv(inter_data_file, index=False)

    return


def main(args, logging):
    '''
    Changes raw data to data to be used for training

    Parameters:
        args(argparse.Namespace) Contains the settings used
        logger(logging.Logger) Used for logging
    '''

    # Change the args to a dict (When processing data a yaml with dict will be used)
    setting = vars(args)

    # Process raw data to data that is used for training
    process_data(setting, logging)

    return


if __name__ == '__main__':
	# Setup argparse to get settings
    parser = argparse.ArgumentParser(description="Set settings for crawling problems.")
    parser.add_argument(
        "-l", "--logging", type=str, default="DEBUG", help="Set logging level."
    )
    parser.add_argument(
        "-r",
        "--raw_data_folder_path",
        type=str,
        default="raw_data",
        help="Path to the raw data folder.",)
    parser.add_argument(
        "-u",
        "--user_data_file",
        type=str,
        default="user_data.csv",
        help="Name of the user data file in the folder.",
    )
    parser.add_argument(
        "-i",
        "--item_data_file",
        type=str,
        default="problem_data.csv",
        help="Name of the item data file in the folder.",
    )
    parser.add_argument(
        "-t",
        "--inter_data_file",
        type=str,
        default="interaction_data.csv",
        help="Name of the inter data file in the folder.",
    )
    parser.add_argument(
        "-p",
        "--processed_data_folder_path",
        type=str,
        default="processed_data",
        help="Path to the output processed data folder.",)

    # Get arguments
    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    # Setup logger
    if not os.path.isdir('log'):
        os.mkdir('log')
    
    logging.basicConfig(
        filename=os.path.join("log", f"data_processing_{start_time}.txt"),
        filemode="a",
        format="%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG
    )

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(args.logging)

    logger.debug("Starting Program")

    main(args, logger)

    logger.debug("Ending Program")

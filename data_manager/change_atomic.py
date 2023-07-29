import argparse
import datetime
import logging
import pandas as pd
import os
import yaml


def create_atomic(setting: dict, logging: logging.Logger) -> None:
    '''
    Changes processed data to atomic data files to be used in recbole

    Parameters:
        setting(dict) Contains the settings used
        logger(logging.Logger) Used for logging
    '''

    from_data_path = setting['path']['processed_data_folder_path']

    logging.debug('Importing data')

    # Get processed data
    from_user_df = pd.read_csv(os.path.join(from_data_path, setting['path']['user_data_file']))
    from_item_df = pd.read_csv(os.path.join(from_data_path, setting['path']['item_data_file']))
    from_inter_df = pd.read_csv(os.path.join(from_data_path, setting['path']['inter_data_file']))
    
    # Change column name to fit atomic file
    rename_user_dict = {i: f"{i}:{v}" for i, v in setting['atomic_user_dict'].items()}
    rename_item_dict = {i: f"{i}:{v}" for i, v in setting['atomic_item_dict'].items()}
    rename_inter_dict = {i: f"{i}:{v}" for i, v in setting['atomic_inter_dict'].items()}

    logging.debug('Renaming columns')

    to_user_df = from_user_df.rename(columns=rename_user_dict)
    to_item_df = from_item_df.rename(columns=rename_item_dict)
    to_inter_df = from_inter_df.rename(columns=rename_inter_dict)

    logging.debug('Saving data')

    # Save data to atomic folder with the same name and different extensions
    to_data_path = setting['path']['atomic_data_folder_path']

    if not os.path.isdir(to_data_path):
        os.mkdir(to_data_path)

    to_user_df.to_csv(os.path.join(to_data_path, f"{to_data_path}.user"), index=False)
    to_item_df.to_csv(os.path.join(to_data_path, f"{to_data_path}.item"), index=False)
    to_inter_df.to_csv(os.path.join(to_data_path, f"{to_data_path}.inter"), index=False)

    return


def main(setting, logging):
    '''
    Changes raw data to data to be used for training

    Parameters:
        logger(logging.Logger) Used for logging
    '''

    # Change processed data to atomic data that is used for recbole
    create_atomic(setting, logging)
    
    return


if __name__ == '__main__':
    # Get the settings
    with open("setting.yaml", "r") as f:
        setting = yaml.load(f, Loader=yaml.FullLoader)
    
    start_time = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    # Setup logger
    if not os.path.isdir('log'):
        os.mkdir('log')
    
    logging.basicConfig(
        filename=os.path.join("log", f"data_atomic_{start_time}.txt"),
        filemode="a",
        format="%(asctime)s %(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG
    )

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    logger.debug("Starting Program")

    main(setting, logger)

    logger.debug("Ending Program")

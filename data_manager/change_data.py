import argparse
import datetime
import logging
import os
import pandas as pd
import yaml


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
    
    # Delete problems in other languages (except KOR, ENG)
    changed_input_df = input_df
    
    result = []
    for index, info in changed_input_df[['problemId','titles']].iterrows():
        pid, titles = info
        title_list = eval(titles)
        for j in title_list:
            language = j['language']
            if language=='ko' or language=='en':
                break
        else:
            result.append(pid)
    changed_input_df=changed_input_df[changed_input_df['problemId'].isin(result)==False]
    
    # Delete problems which 'isSolvable' is False
    changed_input_df=changed_input_df[changed_input_df['isSolvable']==True]
    
    # Adjust the level of problem
    changed_input_df=changed_input_df[(changed_input_df['level']<21) & (changed_input_df['level']>0)]
    
    #Delete problems which 'givesNoRating' is False
    changed_input_df=changed_input_df[changed_input_df['givesNoRating']==False]
    
    # Delete 'tags' value
    changed_input_df=changed_input_df[changed_input_df['tags']!='[]']
    
    print(changed_input_df.shape)
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
    raw_data_folder = setting['path']['raw_data_folder_path']

    # Get path to data
    #user_data_file = os.path.join(raw_data_folder, setting['path']['user_data_file'])
    item_data_file = os.path.join(raw_data_folder, setting['path']['item_data_file'])
    #inter_data_file = os.path.join(raw_data_folder, setting['path']['inter_data_file'])

    logging.debug('Getting raw data from path')

    # Get data
    #user_df = pd.read_csv(user_data_file)
    item_df = pd.read_csv(item_data_file)
    #inter_df = pd.read_csv(inter_data_file)

    logging.debug('Processing user data')

    # Process each data seperatly
    # Process user data
    #processed_user_df = process_user_data(user_df)

    logging.debug('Processing item data')

    # Process item data
    processed_item_df = process_item_data(item_df)

    logging.debug('Processing interaction data')

    # Process interaction data
    #processed_inter_df = process_inter_data(inter_df)

    logging.debug('Processing extra data')

    # Process data together

    logging.debug('Saving processed data')

    # Save processed data
    processed_data_folder = setting['path']['processed_data_folder_path']

    # Create processed data folder
    if not os.path.isdir(processed_data_folder):
        os.mkdir(processed_data_folder)

    #user_data_file = os.path.join(processed_data_folder, setting['path']['user_data_file'])
    item_data_file = os.path.join(processed_data_folder, setting['path']['item_data_file'])
    #inter_data_file = os.path.join(processed_data_folder, setting['path']['inter_data_file'])

    #processed_user_df.to_csv(user_data_file, index=False)
    processed_item_df.to_csv(item_data_file, index=False)
    #processed_inter_df.to_csv(inter_data_file, index=False)

    return


def main(setting, logging):
    '''
    Changes raw data to data to be used for training

    Parameters:
        args(argparse.Namespace) Contains the settings used
        logger(logging.Logger) Used for logging
    '''

    # Process raw data to data that is used for training
    process_data(setting, logging)

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
        filename=os.path.join("log", f"data_processing_{start_time}.txt"),
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

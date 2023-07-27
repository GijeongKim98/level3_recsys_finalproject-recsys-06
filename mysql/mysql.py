import getpass
import os
import pandas as pd
import pymysql
import yaml


PATH = 'setting.yaml'


def connect_db(setting) -> None:
    '''
    Connects to Database

    Parameters:
        setting(dict) Contains the settings used
            host(string) Host name of the database
            port(int) Port of the hosted database
            user(string) Username of the user
            password(string) Password of the user
    Returns:
        connection: Connected database
    '''
    
    if 'password' not in setting:
        print('No Password Detected')
        return
    else:
        password = setting['password']
    
    if 'host' not in setting:
        host = 'database-1.ctbbcoxcjq1p.ap-southeast-2.rds.amazonaws.com'
    else:
        host = setting['host']
    
    if 'port' not in setting:
        port = 3306
    else:
        port = setting['port']
        
    if 'user' not in setting:
        user = 'admin'
    else:
        user = setting['user']
    
    # Connect to database
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    
    return connection


def create_tables(cursor):
    '''
    Creates the tables used

    Parameters:
        cursor: Cursor connected database
    '''
    
    cursor.execute(
        """
            create table problem(
                problem_id int primary key,
                correct_users int,
                level TINYINT,
                voted_users int,
                sprout bool,
                average_tries float
            );
        """
    )

    cursor.execute(
        """
            create table problem_btag(
                problem_id int primary key,
                basics TINYINT, bruteforce TINYINT,
                datastructure TINYINT, dp TINYINT,
                graph TINYINT, greedy TINYINT, 	hash TINYINT, math TINYINT,
                recursion TINYINT, search TINYINT, tree TINYINT,
                foreign key(`problem_id`) references `problem`(`problem_id`)
            );
        """
    )

    cursor.execute(
        """
            create table problem_mtag(
                problem_id int primary key,
                bfs TINYINT, binary_search TINYINT,
                bruteforcing TINYINT, deque TINYINT, dfs TINYINT,
                disjoint_set TINYINT, divide_and_conquer TINYINT,
                dp TINYINT, graphs TINYINT, greedy TINYINT, hashing TINYINT,
                implementation TINYINT, math TINYINT, mitm TINYINT,
                parametric_search TINYINT, priority_queue TINYINT,
                queue TINYINT, recursion TINYINT, segtree TINYINT,
                sorting TINYINT, stack TINYINT, string TINYINT,
                ternary_search TINYINT, tree_set TINYINT, trees TINYINT,
                two_pointer TINYINT,
                foreign key(`problem_id`) references `problem`(`problem_id`)
            );
        """
    )

    cursor.execute(
        """
            create table user(
                user_id VARCHAR(25),
                solved_count int,
                follower_count int,
                following_count int,
                tier TINYINT,
                max_streak int,
                rating double,
                primary key (`user_id`)
            );
        """
    )

    cursor.execute(
        """
            create table user_rating_big(
                user_id VARCHAR(25) primary key,
                tree double, datastructure double, graph double, hash double,
                search double, dp double, greedy double, math double, bruteforce double,
                recursion double, basics double,
                foreign key(`user_id`) references `user`(`user_id`)
            );
        """
    )

    cursor.execute(
        """
            create table user_rating_middle(
                user_id VARCHAR(25) primary key,
                bfs double, binary_search double, bruteforcing double, deque double,
                dfs double, disjoint_set double, divide_and_conquer double, dp double,
                graphs double, greedy double, hashing double, implementation double,
                math double, mitm double, parametric_search double, priority_queue double,
                queue double, recursion double, segtree double,  sorting double, stack double,
                string double, ternary_search double, tree_set double, trees double,
                two_pointer double,
                foreign key(`user_id`) references `user`(`user_id`)
            );
        """
    )
    
    cursor.execute(
        """
            create table company_rating(
                company_id VARCHAR(10) primary key,
                basics double, bruteforce double, datastructure double, dp double,
                graph double, greedy double, hash double, math double,
                recursion double, search double, tree double
            );
        """
    )
    
    cursor.execute(
        """
            create table company_middle_rating(
                company_id VARCHAR(10) primary key,
                implementation double, greedy double, math double, stack double, string double,
                parametric_search double, bfs double, dp double, tree_set double, bruteforcing double,
                binary_search double, disjoint_set double, ternary_search double, two_pointer double,
                deque double, segtree double, graphs double, divide_and_conquer double,
                priority_queue double, queue double, hashing double, trees double, dfs double,
                mitm double, sorting double, recursion double
            );
        """
    )

    
def insert_data(cursor, file_name, table_name, setting):
    '''
    Insers CSV file to a database table

    Parameters:
        cursor: Cursor connected database
        file_name(str): Name of the csv file
        table_name(str): Name of the table to be inserted
        setting(dict) Contains the settings used
            data_path(string) Directory the CSV is in
    '''
    
    temp_df = pd.read_csv(os.path.join(setting['data_path'], file_name))

    if 'joined_date' in temp_df.columns:
        temp_df.drop('joined_date', axis=1, inplace=True)

    temp_iter = len(temp_df) // 1000
    if temp_iter == 0:
        temp_iter = 1
    
    for i in range(len(temp_df)):
        values = str(list(temp_df.iloc[i]))[1:-1]
        cursor.execute(f'insert into {table_name} values ({values});')
        if i % temp_iter == 0:
            print(f'{i / len(temp_df) * 100}% Done')


def main(setting):
    # Connect to database
    connection = connect_db(setting)
    
    # If not connected
    if connection is None:
        return
    
    # Get cursor to execute mysql command
    cursor = connection.cursor()
    
    # Create new database to save data
    if setting['create_db']:
        cursor.execute('create database processed_db')
    cursor.execute('use processed_db;')
    
    connection.commit()
    
    # Create the tables
    if setting['create_table']:
        create_tables(cursor)
    
    connection.commit()
    
    # Insert data to tables
    for temp_file, temp_table in zip(setting['insert_file_list'], setting['insert_table_name']):
        print(f"Working on {temp_table}")
        insert_data(cursor, temp_file, temp_table, setting)
        connection.commit()
        
    # Terminate connection
    cursor.close()
    connection.close()

    
if __name__ == '__main__':
    # load setting
    with open(PATH) as f:
        setting = yaml.load(f, Loader=yaml.FullLoader)
    
    setting['password'] = getpass.getpass(prompt='Enter Password: ')

    main(setting)

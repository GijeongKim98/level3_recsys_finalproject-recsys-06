from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import getpass
import pymysql
import pandas as pd
import os
import numpy as np
import pickle
from torch_geometric.nn.models import LightGCN
import torch


def mysql_connect():
    connection = pymysql.connect(
        host='database-1.ctbbcoxcjq1p.ap-southeast-2.rds.amazonaws.com',
        port=3306,
        user='admin',
        password=password
    )
    
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute('use processed_db;')
    
    return cursor


def get_mysql_query(cursor, input_str):
    try:
        cursor.execute(input_str)
    except pymysql.err.OperationalError as e:
        if e[0] == 2013: # Lost connection to server 
            cursor = mysql_connect()
            cursor.execute(input_str)
        else:
            raise
            
    return cursor


def load_model(model_path, model_name):
    device = "cuda" if torch.cuda.is_available else "cpu"

    model_dict = torch.load(os.path.join(model_path, model_name), map_location=device)

    model = LightGCN(
      num_nodes= model_dict["num_nodes"],
      embedding_dim= model_dict["embedding_dim"],
      num_layers = model_dict["num_layers"]
    )

    model.load_state_dict(model_dict["model_state_dict"])

    model = model.to(device)

    return model


def load_idx2node(data_path, file_name):
    with open(os.path.join(data_path, file_name), "rb") as f:
        idx2node = pickle.load(f)
    return idx2node


class Problem:
    def __init__(self, cursor, problem_id, problem_name_df):
        self.problem_id = problem_id
        
        input_str = f'''\
                select *\
                from problem\
                where problem_id = '{problem_id}';\
            '''
        
        get_mysql_query(cursor, input_str)

        result = cursor.fetchall()[0]

        self.level = result['level']

        self.title = problem_name_df[problem_name_df['problem_id'] == int(problem_id)]['title'].values[0]

        input_str = f'''\
                select *\
                from problem_btag\
                where problem_id = '{problem_id}';\
            '''
        
        get_mysql_query(cursor, input_str)

        result = cursor.fetchall()[0]

        self.bigtag = [k for k, v in result.items() if v == 1]

        input_str = f'''\
                select *\
                from problem_mtag\
                where problem_id = '{problem_id}';\
            '''
        
        get_mysql_query(cursor, input_str)

        result = cursor.fetchall()[0]

        self.midtag = [k for k, v in result.items() if v == 1]

        return
    
class Repository:
    """
    title (str): The title of the repository formed with organization/repo.
    label (int:Categorical): Manually categorized label of the repository.
        Etc.: 0
        Frontend: 1
        Backend: 2
        ML: 3
        Native (Desktop): 4
        Native (Mobile): 5
        Programming Language: 6
        Awesome List: 7
    tags (str:Nullable): Official tags of the repository.
    about (str:Nullable): Description about repository represented in About section.
    readme (str:Nullable): Description from the first paragraph of the readme file.
    lang (str:Categorical, Nullable): Major programming language which compose repository.
    embedding (float: [x, y]): 2D embedding of repository
    """

    def __init__(self, cursor, title):
        self.title = title.replace('_', '/')

        input_str = f'''\
                select *\
                from github_repo\
                where title = '{self.title}';\
            '''
        
        get_mysql_query(cursor, input_str)

        result = cursor.fetchall()[0]
        
        self.tags = result['tags']
        
        self.about = result['about']
        self.readme = result['readme']
        self.lang = result['lang']
        self.embedding = [result['tsne_text_x'], result['tsne_text_y']]
        self.label_28 = result['label_28']


# Get title of problems
problem_name_df = pd.read_csv(os.path.join('data', 'problem_name.csv'))
s_good_problem_df = pd.read_csv('./data/sim_good_problem_ver1.csv')
inter_df = pd.read_csv('./data/interaction_ver3.csv')

lgcn_model = load_model('model', 'lgcn.pt')

general_idx = load_idx2node('data', 'idx2node.pickle')

password = getpass.getpass(prompt='Enter Password: ')

cursor = mysql_connect()

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def main_page():
    return {"This is a API for": "RecSys Group 6"}


#region problem

@app.get("/problem/")
def problem():
    return {"Enter Problem Number": "To Get Problem Data"}

@app.get("/problem/{problem_id}")
def problem(problem_id: str):
    return Problem(cursor, problem_id, problem_name_df)

@app.get("/problem/{problem_id}/stats")
def problem_stats(problem_id: str):
    input_str = f'''\
            select *\
            from problem\
            where problem_id = '{problem_id}';\
        '''
    
    get_mysql_query(cursor, input_str)

    result = cursor.fetchall()[0]

    result['title'] = problem_name_df[problem_name_df['problem_id'] == int(problem_id)]['title'].values[0]

    if type(result) != dict:
        return None

    return result

@app.get("/problem/{problem_id}/btag")
def problem_btag(problem_id: str):
    input_str = f'''\
            select *\
            from problem_btag\
            where problem_id = '{problem_id}';\
        '''
    
    get_mysql_query(cursor, input_str)

    result = cursor.fetchall()

    if type(result) != list:
        return None

    return result[0]

@app.get("/problem/{problem_id}/mtag")
def problem_mtag(problem_id: str):
    input_str = f'''\
            select *\
            from problem_mtag\
            where problem_id = '{problem_id}';\
        '''
    
    get_mysql_query(cursor, input_str)

    result = cursor.fetchall()

    if type(result) != list:
        return None

    return result[0]

@app.get("/problem/{user_id}/{company_str}")
def get_recommend_problems(user_id: str, company_str: str):
    """
    Args:
        user_id: ID of the taget user
        difficulty: Difficulty of the problem relative to the user
                    0: Easy
                    1: Proper
                    2: Challenge
        number: The number of problems to get
    Return:
        problems: Problem list of the target user according to difficulty
    """
    input_str = f'''\
            select *\
            from company_rating\
            where company_id = '{company_str}';\
        '''

    get_mysql_query(cursor, input_str)
    
    result = cursor.fetchall()[0]
    
    company_big_tag = result
    
    input_str = f'''\
            select *\
            from user_rating_big\
            where user_id = '{user_id}';\
        '''
    
    get_mysql_query(cursor, input_str)
    
    result = cursor.fetchall()[0]
    
    user_big_tag = result
    
    solved_problem = list(inter_df[(inter_df['user_id']==user_id)&(inter_df['answer_code']==1)]['problem_id'])
    
    temp_df=s_good_problem_df[s_good_problem_df['problem_id'].isin(solved_problem)==False]
    u_list = np.array([i[1] for i in sorted(list(user_big_tag.items())[1:])])
    c_list = np.array([i[1] for i in sorted(list(company_big_tag.items())[1:])])
    vector = 1-u_list/c_list
    result = []
    for i in range(len(temp_df)):
        result.append([temp_df.iloc[i]['problem_id'], sum(sorted(temp_df.iloc[i][1:-1] * vector)[-2:])+temp_df.iloc[i]['rating']])
    result = [int(i[0]) for i in sorted(result, key=lambda x : x[1])]

    # inference
    
    device='cuda'

    node2idx = {node: idx for idx, node in general_idx.items()}
    edges = []
    for problem_id in result:
        edges.append([node2idx[user_id],node2idx[problem_id]])

    edges = torch.LongTensor(edges).T

    edges = edges.to(device)

    predicts = lgcn_model.predict_link(edge_index=edges, prob=True).detach().cpu().numpy()

    predict_sort = sorted(predicts)

    unit = len(predict_sort) // 10

    q_0, q_1, q_2 = predict_sort[unit], predict_sort[unit*5], predict_sort[unit*9]
    
    hard_problem, medium_problem, easy_problem = [],[],[]

    for idx, pre_dict in enumerate(predicts):
        if pre_dict < q_0:
            continue
        elif pre_dict < q_1 and len(hard_problem) <= 20:
            hard_problem.append(result[idx])
        elif pre_dict < q_2 and len(medium_problem) <= 20:
            medium_problem.append(result[idx])
        elif pre_dict >= q_2 and len(easy_problem) <= 20:
            easy_problem.append(result[idx])

    return hard_problem, medium_problem, easy_problem

#endregion problem


#region user

@app.get("/user/")
def user():
    return {"Enter User Number": "To Get User Data"}

@app.get("/user/{user_id}")
def user_user_id(user_id: str):
    """
    Args:
        user_id: ID of the target user
    Return:
        ratings: Bigtag rating list of the target user
    """
    
    input_str = f'''\
            select *\
            from user\
            where user_id = '{user_id}';\
        '''
    
    get_mysql_query(cursor, input_str)

    result = cursor.fetchall()[0]

    if type(result) != dict:
        return None

    return result

@app.get("/user/{user_id}/btag")
def get_user_bigtag_ratings(user_id: str):
    """
    Args:
        user_id: ID of the target user
    Return:
        ratings: Bigtag rating list of the target user
    """

    input_str = f'''\
            select *\
            from user_rating_big\
            where user_id = '{user_id}';\
        '''
    
    get_mysql_query(cursor, input_str)

    result = cursor.fetchall()[0]

    if type(result) != dict:
        return None

    return result

@app.get("/user/{user_id}/mtag")
def get_user_middletag_ratings(user_id: str):
    """
    Args:
        user_id: ID of the target user
    Return:
        ratings: Middletag rating list of the target user
    """

    input_str = f'''\
            select *\
            from user_rating_middle\
            where user_id = '{user_id}';\
        '''
    
    get_mysql_query(cursor, input_str)

    result = cursor.fetchall()[0]

    if type(result) != dict:
        return None

    return result

#endregion user


@app.get("/companies/")
def get_companies() -> List[str]:
    """
    Return:
        companies: All of the company list
    """

    input_str = f'''\
            select company_id\
            from company_rating;\
        '''

    get_mysql_query(cursor, input_str)

    result = cursor.fetchall()

    return [v['company_id'] for v in result]


@app.get("/companies/{company}/btag")
def get_company_bigtag_ratings(company: str):
    """
    Args:
        company: The name of the target company
    Return:
        ratings: Bigtag rating list of the target company
    """

    input_str = f'''\
            select *\
            from company_rating\
            where company_id = '{company}';\
        '''
    
    get_mysql_query(cursor, input_str)
    
    result = cursor.fetchall()[0]
    del(result['company_id'])

    return result


@app.get("/companies/{company}/mtag")
def get_company_bigtag_ratings(company: str):
    """
    Args:
        company: The name of the target company
    Return:
        ratings: Bigtag rating list of the target company
    """

    input_str = f'''\
            select *\
            from company_middle_rating\
            where company_id = '{company}';\
        '''
    
    get_mysql_query(cursor, input_str)
    
    result = cursor.fetchall()[0]
    del(result['company_id'])

    return result


@app.get("/repo/{title}")
def get_company_bigtag_ratings(title: str):
    """
    Args:
        company: The name of the target company
    Return:
        ratings: Bigtag rating list of the target company
    """
    return Repository(cursor, title)


@app.get("/repo/label/{label}")
def get_company_bigtag_ratings(label: str):
    """
    Args:
        company: The name of the target company
    Return:
        ratings: Bigtag rating list of the target company
    """
    input_str = f'''\
            select *\
            from github_repo\
            where label_28 = '{label}';\
        '''
    
    get_mysql_query(cursor, input_str)
    
    result = cursor.fetchall()
    
    if len(result) > 20:
        result = result[:20]
    
    return result

@app.get("/repo/language/{language}")
def get_company_bigtag_ratings(language: str):
    """
    Args:
        company: The name of the target company
    Return:
        ratings: Bigtag rating list of the target company
    """
    input_str = f'''\
            select *\
            from github_repo\
            where lang = '{language}';\
        '''
    
    get_mysql_query(cursor, input_str)
    
    result = cursor.fetchall()
    
    if len(result) > 20:
        result = result[:20]
    
    return result

@app.get("/repo/search/{label}/{language}")
def get_company_bigtag_ratings(label: str, language: str):
    """
    Args:
        company: The name of the target company
    Return:
        ratings: Bigtag rating list of the target company
    """
    input_str = f'''
            select *
            from github_repo
            where label_28 = '{label}' and lang = '{language}';
        '''
    
    get_mysql_query(cursor, input_str)
    
    result = cursor.fetchall()
    
    if len(result) > 20:
        result = result[:20]
    
    return result

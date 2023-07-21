import argparse
import datetime
import logging
import os
import pandas as pd
import yaml

def process_item(pdf: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes raw item data to data to be used for training

    Parameters:
        pdf(pd.DataFrame) Raw item data
    Returns:
        pdf(pd.DataFrame) Processed problem(item) data
        tag(pd.DataFrame) Raw tag data
        tag_problem(dict) tag별 문제들
    '''
    # 한국어/영어 문제 외 삭제
    result = []
    for index, info in pdf[['problemId','titles']].iterrows():
        pid, titles = info
        title_list = eval(titles)
        for j in title_list:
            language = j['language']
            if language=='ko' or language=='en':
                break
        else:
            result.append(pid)
    pdf=pdf[pdf['problemId'].isin(result)==False]
    
    pdf=pdf[pdf['isSolvable']==True] # isSolvble이 False인 문제 제거
    pdf=pdf[(pdf['level']<21) & (pdf['level']>0)] # level 20초과, unrated 문제 제거
    pdf=pdf[pdf['givesNoRating']==False]
    pdf=pdf[pdf['tags']!='[]'] # tag없는 문제 삭제 (1)
    
    
    # tag column를 tags.csv로 따로 분리
    tag = pd.DataFrame(columns=['tag_id','key','name'])
    tag_set=set()
    tag_problem=dict()
    for i in pdf[['problemId','tags']].iterrows():
        tag_list=eval(i[1]['tags'])
        for j in tag_list:
            key = j['key']
            if key not in tag_problem:
                tag_problem[key]=set()
            tag_problem[key].add(i[1]['problemId'])
            if key in tag_set:
                continue
            tag_set.add(key)
            line = [len(tag_set),key]
            for k in j['displayNames']:
                if k['language']=='ko':
                    line.append(k['name'])
                    break
            tag = pd.concat([tag, pd.DataFrame(line, index=tag.columns, columns=[len(tag)]).T])
    return pdf, tag, tag_problem
    

def process_tag(tag: pd.DataFrame, tag_problem: dict, log_file: str) -> pd.DataFrame:
    '''
    Changes raw tag data to filtered and rearranged several data to be used for training

    Parameters:
        tag(pd.DataFrame) Raw tag data
        tag_problem(dict) tag별 문제들
        log_file log.txt 위치
    Returns:
        final_tag_df(pd.DataFrame) Processed tag data
        problem_btag(pd.DataFrame) Processed big_tag + problem
        problem_mtag(pd.DataFrame) Processed middle_tag + problem
    '''
    
    # log.txt 이용하여 tag filtering
    log=[]
    with open(log_file,'r') as f:
        for line in f.read().replace('\n',' ').split('target : ')[1:]:
            if '(' in line:
                line = line[:line.find('(')]+line[line.find(')')+1:]
            else:
                line
            log.append(line)
    log=list(map(lambda x : x.split(), log))
    log=[line for line in log if line[-1]!='4']
    for line in log:
        key = line[0]
        if key not in tag_problem:
            continue
        if len(line)>2:
            for key2 in line[2:-1]:
                tag_problem[key2] |= tag_problem[key]
        tag_problem.pop(key)
    
    # big_tag 만들기
    middle_tag_problem = tag_problem
    big_tag_problem=dict()
    big_tag_problem['tree'] = middle_tag_problem['trees']

    big_tag_problem['datastructure'] = middle_tag_problem['segtree']
    big_tag_problem['datastructure'] |= middle_tag_problem['stack']
    big_tag_problem['datastructure'] |= middle_tag_problem['priority_queue']
    big_tag_problem['datastructure'] |= middle_tag_problem['queue']
    big_tag_problem['datastructure'] |= middle_tag_problem['priority_queue']
    big_tag_problem['datastructure'] |= middle_tag_problem['deque']
    big_tag_problem['datastructure'] |= middle_tag_problem['disjoint_set']
    big_tag_problem['datastructure'] |= middle_tag_problem['two_pointer']

    big_tag_problem['graph'] = middle_tag_problem['graphs']
    big_tag_problem['graph'] |= middle_tag_problem['bfs']
    big_tag_problem['graph'] |= middle_tag_problem['dfs']

    big_tag_problem['hash'] = middle_tag_problem['tree_set']
    big_tag_problem['hash'] |= middle_tag_problem['hashing']

    big_tag_problem['search'] = middle_tag_problem['binary_search']
    big_tag_problem['search'] |= middle_tag_problem['ternary_search']
    big_tag_problem['search'] |= middle_tag_problem['parametric_search']

    big_tag_problem['dp'] = middle_tag_problem['dp']
    big_tag_problem['greedy'] = middle_tag_problem['greedy']
    big_tag_problem['math'] = middle_tag_problem['math']

    big_tag_problem['bruteforce'] = middle_tag_problem['bruteforcing']
    big_tag_problem['bruteforce'] |= middle_tag_problem['mitm']

    big_tag_problem['recursion'] = middle_tag_problem['recursion']
    big_tag_problem['recursion'] |= middle_tag_problem['divide_and_conquer']

    big_tag_problem['basics'] = middle_tag_problem['sorting']
    big_tag_problem['basics'] |= middle_tag_problem['implementation']
    big_tag_problem['basics'] |= middle_tag_problem['string']
    
    final_tag_df = tag[tag['key'].isin(middle_tag_problem.keys())]
    final_tag_df['big_tag']=0

    final_tag_df.loc[final_tag_df[final_tag_df['key']=='trees'].index,'big_tag']='tree'

    final_tag_df.loc[final_tag_df[final_tag_df['key']=='segtree'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='stack'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='priority_queue'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='queue'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='priority_queue'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='deque'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='disjoint_set'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='two_pointer'].index,'big_tag']='datastructure'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='graphs'].index,'big_tag']='graph'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='bfs'].index,'big_tag']='graph'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='dfs'].index,'big_tag']='graph'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='tree_set'].index,'big_tag']='hash'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='hashing'].index,'big_tag']='hash'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='binary_search'].index,'big_tag']='search'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='ternary_search'].index,'big_tag']='search'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='parametric_search'].index,'big_tag']='search'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='dp'].index,'big_tag']='dp'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='greedy'].index,'big_tag']='greedy'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='math'].index,'big_tag']='math'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='bruteforcing'].index,'big_tag']='bruteforce'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='mitm'].index,'big_tag']='bruteforce'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='recursion'].index,'big_tag']='recursion'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='divide_and_conquer'].index,'big_tag']='recursion'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='sorting'].index,'big_tag']='basics'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='implementation'].index,'big_tag']='basics'
    final_tag_df.loc[final_tag_df[final_tag_df['key']=='string'].index,'big_tag']='basics'
    
    final_tag_df['tag_id']=range(len(final_tag_df))
    final_tag_df=final_tag_df.reset_index().drop('index',axis=1)
    
    
    # middle tag 생성
    tmp=pd.DataFrame()
    for k,v in middle_tag_problem.items():
        for pid in v:
            tmp=pd.concat([tmp, pd.DataFrame([pid, k]).T])
    tmp.columns=['problem_id','middle_tag']
    tmp.sort_values('problem_id',inplace=True)
    tmp.index=range(len(tmp))
    problem_mtag = pd.get_dummies(tmp, columns=['middle_tag'],prefix_sep='',prefix='').groupby('problem_id').sum().reset_index()
    
    # big tag 생성
    tmp=pd.DataFrame()
    for k,v in big_tag_problem.items():
        for pid in v:
            tmp=pd.concat([tmp, pd.DataFrame([pid, k]).T])
    tmp.columns=['problem_id','big_tag']
    tmp.sort_values('problem_id',inplace=True)
    tmp.index=range(len(tmp))
    problem_btag = pd.get_dummies(tmp, columns=['big_tag'],prefix_sep='',prefix='').groupby('problem_id').sum().reset_index()

    return final_tag_df, problem_mtag, problem_btag

def process_ui_data(udf: pd.DataFrame, idf: pd.DataFrame, pdf: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes raw user data, interaction data to process data to be used for training

    Parameters:
        udf(pd.DataFrame) Raw user data
        idf(pd.DataFrame) Raw interaction data
        pdf(pd.DataFreme) preprocessed problem data
    Returns:
        udf(pd.DataFreme) Processed user data
        ddf(pd.DataFreme) Processed interaction data
    '''
    
    # 중복 유저 drop
    udf = udf.sort_values('solvedCount')
    udf = udf[~udf['handle'].duplicated(keep='last')]
    
    
    udf=udf[udf['solvedCount']!=0] # 푼 문제가 없는 유저 삭제
    udf=udf[udf['rating']!=0] # rating이 0인 유저 삭제
    udf = udf[udf['proUntil']!='9999-12-31T00:00:00.000Z'] # 관리자 제거
    
    # User Column drop & rename
    udf.drop(['bio', 'badgeId', 'backgroundId', 'profileImageUrl', 'voteCount', 'class', 'classDecoration', 'rating', 'ratingByProblemsSum','ratingByClass', 
              'ratingBySolvedCount', 'ratingByVoteCount','coins', 'stardusts', 'bannedUntil','proUntil','rank', 'isRival', 'isReverseRival'], inplace=True, axis=1)
    udf.rename(columns={"handle":'user_id','solvedCount':'solved_count','rivalCount':'follower_count','reverseRivalCount':'following_count',
                        'joinedAt':'joined_date','maxStreak':'max_streak'},inplace=True)
    
    # Interaction: level 추가 및 problem data에 없는 문제 제거
    idf.columns=['user_id','problem_id','answer_code']
    idf = pd.merge(idf, pdf[['problem_id','level']], how='right')
    idf = idf.sort_values(['user_id','level','problem_id','answer_code'])
    idf = idf.sort_values('answer_code')
    idf = idf[~idf.duplicated(keep='last')]
    idf.dropna(inplace=True)
    
    # user_id 통일하기
    user_list = set(idf['user_id'].unique()) & set(udf['user_id'].unique())
    idf=idf[idf['user_id'].isin(user_list)]
    udf=udf[udf['user_id'].isin(user_list)]
    
    # user에 solved_count 추가(intraction 반영)
    tmp = idf[['user_id','answer_code']].groupby('user_id').sum()
    assert len(udf)==len(tmp)
    udf.sort_values('user_id', inplace=True)
    udf['solved_count'] = list(tmp['answer_code'])
    
    # user, interaction에서 푼 문제가 없는 유저 제거(solved_count 반영)
    tmp_user=udf[udf['solved_count']==0]['user_id']
    udf = udf[udf['user_id'].isin(tmp_user)==False]
    idf = idf[idf['user_id'].isin(tmp_user)==False]
    
    return udf, idf

def process_rating_data(problem_mtag: pd.DataFrame, problem_btag: pd.DataFrame, pdf: pd.DataFrame, idf: pd.DataFrame) -> pd.DataFrame:
    '''
    Create middle_tag_data
    
    Parameters:
        problem_mtag(pd.DataFrame) problem's middle tag data
        problem_btag(pd.DataFrame) problem's middle tag data
        pdf(pd.DataFreme) preprocessed problem data
        idf(pd.DataFrame) preprocessed interaction data
    Returns:
        user_rating_middle(pd.DataFreme) Processed user's middle_tag rating data
        user_rating_big(pd.DataFreme) Processed user's big_tag rating data
    '''
    
    
    tmp = pd.merge(problem_mtag, pdf[['problem_id','level']])
    for col in tmp.columns[1:-1]:
        tmp[col] *= tmp['level']**(0.9 +0.02*tmp['level'])
    tmp=pd.merge(idf[idf['answer_code']==1], tmp, on='problem_id')
    tmp.drop(['problem_id','answer_code','level_y','level_x'],inplace=True, axis=1)
    def top(df, n=30):
        tag_list = [df[tag_name].sort_values(ascending=False)[:n].sum() for tag_name in problem_mtag.columns[1:]]
        return pd.DataFrame(tag_list,index=problem_mtag.columns[1:]).T
    user_rating_middle = tmp.groupby('user_id').apply(top)
    user_rating_middle=user_rating_middle.reset_index().drop('level_1',axis=1).set_index('user_id')
    
    user_rating_big=pd.DataFrame()
    user_rating_big['tree'] = user_rating_middle['trees']

    user_rating_big['datastructure'] = user_rating_middle['segtree']
    user_rating_big['datastructure'] += user_rating_middle['stack']
    user_rating_big['datastructure'] += user_rating_middle['priority_queue']
    user_rating_big['datastructure'] += user_rating_middle['queue']
    user_rating_big['datastructure'] += user_rating_middle['priority_queue']
    user_rating_big['datastructure'] += user_rating_middle['deque']
    user_rating_big['datastructure'] += user_rating_middle['disjoint_set']
    user_rating_big['datastructure'] += user_rating_middle['two_pointer']

    user_rating_big['graph'] = user_rating_middle['graphs']
    user_rating_big['graph'] += user_rating_middle['bfs']
    user_rating_big['graph'] += user_rating_middle['dfs']

    user_rating_big['hash'] = user_rating_middle['tree_set']
    user_rating_big['hash'] += user_rating_middle['hashing']

    user_rating_big['search'] = user_rating_middle['binary_search']
    user_rating_big['search'] += user_rating_middle['ternary_search']
    user_rating_big['search'] += user_rating_middle['parametric_search']

    user_rating_big['dp'] = user_rating_middle['dp']
    user_rating_big['greedy'] = user_rating_middle['greedy']
    user_rating_big['math'] = user_rating_middle['math']

    user_rating_big['bruteforce'] = user_rating_middle['bruteforcing']
    user_rating_big['bruteforce'] += user_rating_middle['mitm']

    user_rating_big['recursion'] = user_rating_middle['recursion']
    user_rating_big['recursion'] += user_rating_middle['divide_and_conquer']

    user_rating_big['basics'] = user_rating_middle['sorting']
    user_rating_big['basics'] += user_rating_middle['implementation']
    user_rating_big['basics'] += user_rating_middle['string']
    
    return user_rating_middle, user_rating_big
    

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
    user_data_file = os.path.join(raw_data_folder, setting['path']['user_data_file'])
    item_data_file = os.path.join(raw_data_folder, setting['path']['item_data_file'])
    inter_data_file = os.path.join(raw_data_folder, setting['path']['inter_data_file'])
    log_file = setting['path']['log_file']

    logging.debug('Getting raw data from path')

    # Get data
    item_df = pd.read_csv(item_data_file)
    user_df = pd.read_csv(user_data_file)
    inter_df = pd.read_csv(inter_data_file)
    
    
    
    logging.debug('Processing item data')
    
    pdf, tag,tag_problem = process_item(item_df)
    final_tag_df, problem_mtag, problem_btag = process_tag(tag, tag_problem, log_file)
    
    pdf=pdf[pdf['problemId'].isin(problem_mtag['problem_id'])==True] # tag없는 문제 삭제 (2)
    
    # column drop & rename
    pdf.drop(['titleKo','titles','isSolvable','isPartial','givesNoRating','isLevelLocked','official','tags','metadata'], inplace=True, axis=1)
    pdf.rename(columns={"problemId":'problem_id','acceptedUserCount':'correct_users','votedUserCount':'voted_users','averageTries':'average_tries'},inplace=True)
    
    
    logging.debug('Processing user & interaction data')
    
    # Process User data(1) & Process interaction data
    udf, idf = process_ui_data(user_df, inter_df, pdf)

    # user_rating_middle, user_rating_big 생성
    user_rating_middle,user_rating_big = process_rating_data(problem_mtag,problem_btag, pdf, idf)
    
    # Process user data(2)
    # rating 추가 (user_rating_big/middle 반영)
    udf.sort_values('user_id',inplace=True)
    udf['rating'] = list(user_rating_big.sum(axis=1))
    udf.index=range(len(udf))


    logging.debug('Saving processed data')

    # Save processed data
    processed_data_folder = setting['path']['processed_data_folder_path']

    # Create processed data folder
    if not os.path.isdir(processed_data_folder):
        os.mkdir(processed_data_folder)

    user_data_file = os.path.join(processed_data_folder, setting['path']['user_data_file'])
    item_data_file = os.path.join(processed_data_folder, setting['path']['item_data_file'])
    inter_data_file = os.path.join(processed_data_folder, setting['path']['inter_data_file'])
    tag_data_file = os.path.join(processed_data_folder, setting['path']['tag_data_file'])
    problem_big_tag_file= os.path.join(processed_data_folder, setting['path']['problem_big_tag_file'])
    problem_middle_tag_file= os.path.join(processed_data_folder, setting['path']['problem_middle_tag_file'])
    user_rating_middle_file= os.path.join(processed_data_folder, setting['path']['user_rating_middle_file'])
    user_rating_big_flie= os.path.join(processed_data_folder, setting['path']['user_rating_big_flie'])

    udf.to_csv(user_data_file, index=False)
    pdf.to_csv(item_data_file, index=False)
    idf.to_csv(inter_data_file, index=False)
    final_tag_df.to_csv(tag_data_file, index=False)
    problem_mtag.to_csv(problem_middle_tag_file, index=False)
    problem_btag.to_csv(problem_big_tag_file, index=False)
    user_rating_middle.to_csv(user_rating_middle_file)
    user_rating_big.to_csv(user_rating_big_flie)

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

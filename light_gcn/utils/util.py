import pandas as pd
import numpy as np
import os
import yaml
import json
import pickle
from typing import List
from datetime import datetime
import torch
import random


def load_data(data_path, file_name):
    data = dict()
    data["item"] = pd.read_csv(os.path.join(data_path, file_name["item"]))
    data["user"] = pd.read_csv(os.path.join(data_path, file_name["user"]))
    data["inter"] = pd.read_csv(os.path.join(data_path, file_name["interaction"]))
    return data


def load_setting(setting_path):
    with open(setting_path, "r") as f:
        setting = yaml.load(f, Loader=yaml.FullLoader)
    return setting


def init():
    setting_path = os.path.join(os.getcwd(), "utils/setting.yaml")
    setting = load_setting(setting_path)
    data = load_data(setting["path"]["data_path"], setting["file_name"])
    logger = get_logger(logging_conf)

    setting["now_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    if setting["is_train"]:
        setting["file_name"]["save_model"] = (
            setting["model_name"] + "_" + setting["now_time"] + ".pt"
        )

        setting["file_name"]["idx2node"] = "idx2node_" + setting["now_time"] + ".pickle"

    setting["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    seed = setting["seed"]
    # All results will be fixed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return setting, data, logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {  # formatters => Basic 사용
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },  # 로그 작성 시간, 작성 이름, 레벨(Info, Debug 등), 로그메세지
    "handlers": {  # 로그 출력 방식
        "console": {  # 터미널
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {  # 파일
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "logs/run4.log",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file_handler"]},
}


def get_logger(logger_conf: dict):
    import logging
    import logging.config

    if not os.path.isdir(os.getcwd() + "/logs"):
        os.mkdir(os.getcwd() + "/logs")

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


# tag별 푼 문제 수 에 대해 개수의 절반을 안푼 문제중 같은 tag의 문제를 추가하기
"""
def negative_sampler(data_inter_by_uid, data_path, negative_sample_prob=0.5, seed=42):
    tag_level_problem = load_problem_by_tag_data(data_path) # tag.json 파일 불러오기 dict type
    problem_set_by_uid = set(data_inter_by_uid["problem_id"]) # user가 푼 문제
    count_problem_by_user = len(problem_set_by_uid) # user가 푼 문제의 수
    constant = 1+ 1/((count_problem_by_user // 100)+1) # 상수항
    np.random.seed(seed)
    
    tag_solved_not_probs = []
    
    for tag_name, problem_set in problem_by_tag.items():
        solve_tag_count = len(eval(problem_set).intersection(problem_set_by_uid))
        tag_solved_not_prob = 1 - (solve_tag_count / count_problem_by_user)
        tag_solved_not_probs.append(tag_solved_not_prob)
    
    sum_ = sum(tag_solved_not_probs)
    tag_solved_not_probs = list(map(lambda x: x/sum_, tag_solved_not_probs))
    
    union_set= set()
    
    for idx, (tag_name, problem_set) in enumerate(problem_by_tag.items()):
        subtract_problem = np.array(list(eval(problem_set) - problem_set_by_uid))
        negative_count = int(count_problem_by_user * tag_solved_not_probs[idx] * negative_sample_prob * constant)
        negative_count = len(subtract_problem) if negative_count > len(subtract_problem) else negative_count
        negative_count = 1 if negative_count < 1 else negative_count
        negative_sample = np.random.choice(
            subtract_problem, 
            negative_count,
            replace=False
            )
        # print(f"{tag_name} negative sample : {negative_sample}\n")
        union_set = union_set.union(set(negative_sample))
    
    # print(f"시도한 문제 개수 : {count_problem_by_user}")
    # print(f"neg 개수 : {len(union_set)}")

    return union_set
"""

# 최대 level
MAX_LEVEL = 20


def negative_sampler(
    data_inter_by_uid,
    problem_by_tag,
    user_tier_dict,
    negative_sample_prob=0.5,
    seed=42,
    d_tier=3,
):
    problem_set_by_uid = set(data_inter_by_uid["problem_id"])  # user가 푼 문제
    count_problem_by_user = len(problem_set_by_uid)  # user가 푼 문제의 수
    constant = 1 + 1 / ((count_problem_by_user // 100) + 1)  # 상수항
    np.random.seed(seed)

    tag_solved_not_probs = []

    for tag_name, problem_set in problem_by_tag.items():
        solve_tag_count = len(
            eval(problem_set[str(MAX_LEVEL)]).intersection(problem_set_by_uid)
        )
        tag_solved_not_prob = 1 - (solve_tag_count / count_problem_by_user)
        tag_solved_not_probs.append(tag_solved_not_prob)

    sum_ = sum(tag_solved_not_probs)
    tag_solved_not_probs = list(map(lambda x: x / sum_, tag_solved_not_probs))

    union_set = set()
    min_tier = user_tier_dict[data_inter_by_uid.iloc[0]["user_id"]] - d_tier

    for idx, (tag_name, problem_level_dict) in enumerate(problem_by_tag.items()):
        lowest_level = 15 if min_tier > 15 else (1 if min_tier <= 0 else min_tier)
        problem_above_lowest_level = eval(problem_level_dict[str(MAX_LEVEL)]) - eval(
            problem_level_dict[str(lowest_level)]
        )
        subtract_problem = np.array(
            list(problem_above_lowest_level - problem_set_by_uid)
        )
        negative_count = int(
            count_problem_by_user
            * tag_solved_not_probs[idx]
            * negative_sample_prob
            * constant
        )
        negative_count = (
            len(subtract_problem)
            if negative_count > len(subtract_problem)
            else negative_count
        )
        negative_count = 1 if negative_count < 1 else negative_count
        negative_sample = np.random.choice(
            subtract_problem, negative_count, replace=False
        )
        # print(f"{tag_name} negative sample : {negative_sample}\n")
        union_set = union_set.union(set(negative_sample))

    # print(f"시도한 문제 개수 : {count_problem_by_user}")
    # print(f"neg 개수 : {len(union_set)}")

    return union_set


def load_problem_by_tag_data(data_path):
    tag_level_path = os.path.join(data_path, "prefix_tag_level_ver2.json")
    with open(tag_level_path, "r") as f:
        problem_by_tag = json.load(f)
    user_tier_path = os.path.join(data_path, "user_tier_dict.json")
    with open(user_tier_path, "r") as f:
        user_tier_dict = json.load(f)
    return problem_by_tag, user_tier_dict


def load_negative_sample_data(path, file_name):
    neg_sample_data = pd.read_csv(os.path.join(path, file_name))
    return neg_sample_data


def load_split_data(path):
    train_df = pd.read_csv(os.path.join(path, "train_df.csv"))
    valid_df = pd.read_csv(os.path.join(path, "valid_df.csv"))
    return train_df, valid_df


def save_idx2node(idx2node: dict, path, file_name):
    with open(os.path.join(path, file_name), "wb") as fw:
        pickle.dump(idx2node, fw)

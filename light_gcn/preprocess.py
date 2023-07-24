import pandas as pd
import numpy as np
import os

from utils.util import (
    negative_sampler,
    load_negative_sample_data,
    load_split_data,
    load_problem_by_tag_data,
    save_idx2node,
)


def preprocess(setting: dict, data: dict, logger) -> dict:
    model_name = setting["model_name"].lower()
    logger.debug(f"Start Preprocess Model : {model_name}")
    if model_name == "lgcn":
        if not setting["negative_sample_exist"]:
            logger.debug("Negative Sample file Create")
            negative_inter_merge = get_neg_samples(data, setting, logger)
        else:
            logger.debug("Negative Sample file load")
            negative_inter_merge = load_negative_sample_data(
                setting["path"]["neg_sample_path"], setting["file_name"]["neg_sample"]
            )
        data = preprocess_lgcn(data, logger)
        data["neg_sample"] = negative_inter_merge

        save_idx2node(
            data["idx2node"],
            setting["path"]["idx2node"],
            setting["file_name"]["idx2node"],
        )
    else:
        raise KeyError(f"{model_name} is not found")

    return data


def get_neg_samples(data: dict, setting: dict, logger) -> dict:
    problem_by_tag, user_tier_dict = load_problem_by_tag_data(
        setting["path"]["data_path"]
    )  # tag.json 파일 불러오기 dict type

    count_of_user = data["inter"]["user_id"].nunique()

    negative_user_ids = []
    negative_problem_ids = []

    logging_count = 0

    pre_idx = 0

    for idx in range(len(data["inter"])):
        if data["inter"].loc[idx]["user_id"] != data["inter"].loc[pre_idx]["user_id"]:
            user_inter_df = data["inter"].loc[pre_idx : idx - 1]
            user_negative_sample = negative_sampler(
                user_inter_df,
                problem_by_tag,
                user_tier_dict,
                setting["negative_sample_prob"],
                setting["seed"],
                setting["delta_tier"],
            )

            count_of_negative_sample = len(user_negative_sample)
            negative_user_ids.extend(
                count_of_negative_sample * [data["inter"].loc[pre_idx]["user_id"]]
            )
            negative_problem_ids.extend(list(user_negative_sample))
            logging_count += 1

            pre_idx = idx

            if logging_count % 100 == 0:
                logger.debug(
                    f"Negative Sampling {logging_count} / {count_of_user} Done"
                )

    negative_inter_merge = pd.DataFrame()
    negative_inter_merge["user_id"] = negative_user_ids
    negative_inter_merge["problem_id"] = negative_problem_ids
    negative_inter_merge["answer_code"] = [0] * len(negative_user_ids)

    problem_level_data = data["item"][["problem_id", "level"]]
    negative_inter_merge = pd.merge(
        negative_inter_merge, problem_level_data, how="inner", on="problem_id"
    )

    negative_inter_merge.to_csv(
        setting["path"]["neg_sample_path"] + "/neg_sample_ver2.csv", index=False
    )

    return negative_inter_merge


def preprocess_lgcn(data: dict, logger):
    preprocessed_data = dict()

    preprocessed_data["train"] = data["inter"][data["inter"]["answer_code"] >= 0]
    preprocessed_data["test"] = data["inter"][data["inter"]["answer_code"] < 0]
    preprocessed_data["node2idx"], preprocessed_data["idx2node"] = make_node(data)
    preprocessed_data["num_nodes"] = len(preprocessed_data["node2idx"])

    return preprocessed_data


def make_node(data: dict) -> dict:
    user_id, item_id = (
        sorted(data["inter"]["user_id"].unique().tolist()),
        sorted(data["item"]["problem_id"].unique().tolist()),
    )
    # merge user_id & item_id
    node_id = user_id + item_id

    # Initialization dictionary : node2idx
    node2idx = {node: idx for idx, node in enumerate(node_id)}
    idx2node = {idx: node for node, idx in node2idx.items()}

    return node2idx, idx2node


def split(setting: dict, data: dict, logger) -> dict:
    model_name = setting["model_name"].lower()
    if model_name == "lgcn":
        if not setting["train_valid_data_exist"]:
            logger.info(f"Start Data Split [Model : {model_name}]")
            data["train"], data["valid"] = split_lgcn(
                data, setting["train_valid_prob"], logger, setting["path"]["data_path"]
            )
        else:
            logger.info(f"Load split data [Model : {model_name}]")
            data["train"], data["valid"] = load_split_data(setting["path"]["data_path"])

    return data


def split_lgcn(data: dict, train_valid_prob, logger, save_path):
    pre_idx = 0
    logging_count = 0
    number_of_user = data["train"]["user_id"].nunique()

    valid_idx_set = set()

    for idx in range(len(data["train"])):
        if data["train"].iloc[idx]["user_id"] != data["train"].iloc[pre_idx]["user_id"]:
            number_of_inter = idx - pre_idx
            split_point = int(number_of_inter * train_valid_prob)
            valid_idx_set = valid_idx_set.union(
                {k for k in range(pre_idx + split_point, idx)}
            )
            pre_idx = idx
            logging_count += 1

            if logging_count % 100 == 0:
                logger.debug(f"Split : {logging_count} / {number_of_user} Done")

    train_idx_set = {k for k in range(len(data["train"]))} - valid_idx_set
    train_df = data["train"][data["train"].index.isin(train_idx_set)]
    train_df = pd.concat([train_df, data["neg_sample"]]).sort_values(
        by=["user_id", "level"]
    )

    valid_df = data["train"][data["train"].index.isin(valid_idx_set)]

    train_df.to_csv(save_path + "/train_df.csv")
    valid_df.to_csv(save_path + "/valid_df.csv")

    return train_df, valid_df

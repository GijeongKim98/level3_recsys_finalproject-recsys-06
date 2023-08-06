import os
import time
import logging
import argparse
import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

"""
    title (str): The title of the repository formed with organization/repo.
    tags (str:Nullable): Official tags of the repository.
    about (str:Nullable): Description about repository represented in About section.
    readme (str:Nullable): Description from the first paragraph of the readme file.
    lang (str:Categorical, Nullable): Major programming language which compose repository.
"""
COLUMNS = ["title", "tags", "about", "readme", "lang"]

RANK_BASE_URL = "https://gitstar-ranking.com/repositories"
RANK_SELECTORS = {"repo": "list-group-item paginated_item"}

BASE_URL = "https://github.com/"
SELECTORS = {
    "tags": "topic-tag topic-tag-link",
    "about": "#repo-content-pjax-container > div > div > div.Layout.Layout--flowRow-until-md.Layout--sidebarPosition-end.Layout--sidebarPosition-flowRow-end > div.Layout-sidebar > div > div.BorderGrid-row.hide-sm.hide-md > div > p",
    "readme": "Box-body px-5 pb-5",
    "lang": "color-fg-default text-bold mr-1",
}


def get_parser_from_ranking(page_num: int) -> Optional[BeautifulSoup]:
    query = {"page": page_num}
    response = requests.get(RANK_BASE_URL, params=query)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        return soup
    else:
        print(response.status_code)
        return None


def get_repos_from_parser(parser: BeautifulSoup) -> List[str]:
    repo_with_selectors = parser.find_all("a", RANK_SELECTORS["repo"])
    repos = []
    for r in repo_with_selectors:
        repo = r.get("href")[1:]
        repos.append(repo)
    return repos


def get_parser_from_repo(repo: str) -> Optional[BeautifulSoup]:
    url = BASE_URL + repo
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        return soup
    else:
        print(response.status_code)
        return None


def get_about_from_parser(parser: BeautifulSoup) -> Optional[str]:
    about = parser.select_one(SELECTORS["about"])
    try:
        about_txt = about.get_text().strip()
        return about_txt
    except:
        return None


def get_readme_from_parser(parser: BeautifulSoup) -> Optional[str]:
    readme_contents = parser.find("div", SELECTORS["readme"])
    try:
        readme_paragraphs = readme_contents.find_all("p")
        for p in readme_paragraphs:
            readme_firt_paragraph = p.get_text().strip()
            if readme_firt_paragraph == "":
                continue
            else:
                return readme_firt_paragraph
        return readme_firt_paragraph
    except:
        return None


def get_lang_from_parser(parser: BeautifulSoup) -> Optional[str]:
    try:
        lang = parser.find("span", SELECTORS["lang"]).get_text().strip()
        return lang
    except:
        return None


def get_columns_from_repo(repo: str) -> pd.Series:
    parser = get_parser_from_repo(repo)
    if parser is None:
        print(f"Error with parsing repository: {repo}")
        return None
    else:
        row = pd.Series(
            {
                "title": repo,
                "tags": get_tags_from_parser(parser),
                "about": get_about_from_parser(parser),
                "readme": get_readme_from_parser(parser),
                "lang": get_lang_from_parser(parser),
            }
        )
        return row


def get_tags_from_parser(parser: BeautifulSoup) -> Optional[str]:
    tags_with_selectors = parser.find_all("a", SELECTORS["tags"])
    tags = []
    for t in tags_with_selectors:
        tags.append(t.get_text().strip())
    return tags


def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    # Create saved data folder if it doesn't exist
    if not os.path.isdir("data"):
        logger.info("Data directory missing...\nCreating data directory")
        os.mkdir("data")
        logger.debug("Data directory created")

    if not os.path.exists(os.path.join("data", f"{args.data}")):
        logger.warning("User data file not found")
        return

    repos_df = pd.read_csv(f"data/{args.data}")
    start_index = args.start
    end_index = args.end

    logger.debug("Starting crawling")
    repos = repos_df["title"].to_list()

    rows = []
    start_time = datetime.datetime.now()
    for i, repo in enumerate(repos[start_index : end_index + 1]):
        logging.info(f"{i}: {repo}")
        row = get_columns_from_repo(repo)
        if row is None:
            continue
        rows.append(row)
        time.sleep(4)
    end_time = datetime.datetime.now()

    repo_info_df = pd.DataFrame.from_records(rows)
    repo_info_df.to_csv(f"data/repo_info_{start_index}_{end_index}.csv", index=False)
    logging.info(
        f"{str(end_time-start_time)}: Save repo_info_{start_index}_{end_index}.csv."
    )


if __name__ == "__main__":
    # Setup argparse to get settings
    parser = argparse.ArgumentParser(
        description="Set settings for crawling github repos."
    )
    parser.add_argument(
        "-l", "--logging", type=str, default="INFO", help="Set logging level."
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="popular_repos.csv",
        help="Data file contains the name of the repositories to crawl.",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="The start of the repo index.",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=4999,
        help="The end of the repo index.",
    )
    # Get arguments
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(args.logging)

    logger.debug("Starting Program")

    main(args, logger)

    logger.debug("Ending Program")

import os
import pandas as pd
import xml.etree.ElementTree as et
import glob

from preprocess import *


def get_pandas_from_xml(xml_path):
    """
    Parses the RET datasets to return a pandas dataframe
    Args:
        xml_path (string): path at which xml file is stored
    Returns:
        out_df (pd.DataFrame): dataframe containing dataset extracted from xml
    """
    xtree = et.parse(xml_path)
    xroot = xtree.getroot()
    df_cols = ["id", "t", "h", "entailment", "task"]
    out_df = pd.DataFrame(columns=df_cols)

    # map labels to desired categories
    threeway_dict = {"YES": "AGREE", "NO": "CONTRADICTS", "UNKNOWN": "UNRELATED"}
    contra_dict = {"YES": "CONTRADICTS", "NO": "UNRELATED"}

    for node in xroot:
        node_t = node.find("t").text
        node_h = node.find("h").text
        node_id = node.attrib.get("id")

        if node.attrib.get("entailment") is not None:
            node_entailement = threeway_dict[node.attrib.get("entailment")]
        else:
            # In the case of using the contradiction dataset
            # In terms of 3-way decision, the contradiction="YES" items
            # should be mapped to entailment="NO",
            # and the contradiction="NO" to entailment="UNKNOWN".
            node_entailement = contra_dict[node.attrib.get("contradiction")]

        if node.attrib.get("task") is not None:
            node_task = node.attrib.get("task")
        elif node.attrib.get("type") is not None:
            node_task = node.attrib.get("type")
        else:
            node_task = "missing"

        out_df = out_df.append(
            pd.Series(
                [node_id, node_t, node_h, node_entailement, node_task], index=df_cols
            ),
            ignore_index=True,
        )

    return out_df


def process_dataset(pd_df):
    """
    Process a pandas dataset and return the pandas df 
    with new columns for features and processed pair
    """
    pd_df["entailment"] = pd_df["entailment"].astype("category")

    proc_pairs = [
        preprocess_pair(t, h) for t, h in zip(list(pd_df["t"]), list(pd_df["h"]))
    ]
    features_pd = pd.DataFrame(proc_pairs, columns=feat_cols)

    proc_pairs_reversed = [
        preprocess_pair(h, t) for t, h in zip(list(pd_df["t"]), list(pd_df["h"]))
    ]
    features_pd_reversed = pd.DataFrame(proc_pairs_reversed, columns=feat_cols)

    pd_df1 = pd.concat([pd_df, features_pd], axis=1, sort=False)
    pd_df2 = pd.concat([pd_df, features_pd_reversed], axis=1, sort=False)

    pd_df = pd.concat([pd_df1, pd_df2], axis=0, sort=False)
    return pd_df


def get_dataset():
    """
    Get a dataset based on all the xml dataset files
    """
    all_files = glob.glob("datasets/*")
    all_dfs = []
    for f in all_files:
        temp_pd = get_pandas_from_xml(f)
        temp_pd = process_dataset(temp_pd)
        all_dfs.append(temp_pd)
    dataset = pd.concat(all_dfs, axis=0, sort=False)
    return dataset

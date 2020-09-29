#!/usr/bin/env python

"""Module and CLI for file metadata extraction"""

import hashlib
import logging
import os
from typing import List, Optional

import pandas as pd  # type: ignore

#logging.basicConfig(level=logging.DEBUG)


def load_files(dir_or_file: str) -> List[str]:
    """
    Loads file from provided directory or list file

    Args:
        dir_or_file (str):  Path to directory or list file

    Returns:
         list[str]: List of paths

    """
    all_files = []
    # if path is directory, load from there
    if os.path.isdir(dir_or_file):
        for root, dirs, files in os.walk(dir_or_file, topdown=True):
            for file in files:
                all_files.append(os.path.join(root, file))
    # treat input as list of file names
    else:
        with open(dir_or_file) as f:
            all_files = f.read().splitlines()
    return all_files


def extract_dir(path: str) -> str:
    """
    Extract whole directory from file path

    Args:
        path (str): Path to file

    Returns:
        str: All directories

    """

    return os.path.split(path)[0]


def extract_last_dir(path: str) -> str:
    """
    Extract last directory from file path

    Args:
        path (str): Path to file

    Returns:
        str: Last directory

    """

    return extract_name(extract_dir(path))


def extract_file(path: str) -> str:
    """
    Extract file from file path

    Args:
        path (str): Path to file

    Returns:
        str: filen

    """

    return os.path.split(path)[1]


def extract_name(path: str) -> str:
    """
    Extract file name from file path

    Args:
        path (str): Path to file

    Returns:
        str: filename without extension

    """

    return os.path.splitext(os.path.split(path)[1])[0]


def extract_extension(path: str) -> str:
    """
    Extract lowercase file extension from file path

    Args:
        path (str): Path to file

    Returns:
        str: file extension
    """
    ext = os.path.splitext(os.path.split(path)[1])[1].lower()
    if ext:
        return ext
    else:
        return "None"


def calculate_checksum(path: str, method: str = 'md5') -> str:
    """
    Calculates checksum of provided path

    Args:
        file_path (str): Path to file
        method (str): Which type of checksum

    Returns:
        str: Checksum of file

    """

    with open(path, 'rb') as f:
        data = f.read()
        h = hashlib.new(method)
        h.update(data)
        checksum = h.hexdigest()
    return checksum


def calculate_checksums(path: str, out_file: str = None) -> Optional[pd.DataFrame]:
    """
    Calculates checksums of files  in a directory of list of files
    If out_file is not provided, returns the dataframe, otherwise writes it.

    Args:
        path (str): Path to file or directory containing files
        out_file (str): Name of output csv. If None, nothing is written

    Returns:
        pd.Dataframe: result dataframe
    """

    file_list = load_files(path)
    df = pd.DataFrame()
    df['file'] = file_list
    df['md5'] = df['file'].apply(calculate_checksum)

    num_docs = len(df)
    num_unique_docs = len(df.groupby('md5'))

    logging.info('Total number of documents: %d' % num_docs)
    logging.info('Number of unique documents: %d' % num_unique_docs)


    if out_file:
        df.to_csv(out_file)
    else:
        return df


def find_duplicates(path: str, out_file: str = None) -> Optional[pd.DataFrame]:
    """
    Find duplicate files in a directory of list of files
    If out_file is not provided, returns the dataframe, otherwise writes it.

    Args:
        path (str): Path to file or directory containing files
        out_file (str): Where to write dataframe

    Returns:
        pd.Dataframe: result dataframe
    """

    df = calculate_checksums(path)
    counts = df['md5'].value_counts()
    dups = counts[counts > 1]
    logging.debug(dups)
    df['duplicate'] = df['md5'].isin(dups.index)
    df = df[df['duplicate']]
    if out_file:
        df.to_csv(out_file)
    else:
        return df

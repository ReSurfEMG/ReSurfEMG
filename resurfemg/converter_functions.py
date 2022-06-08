"""
This file contains functions to work with various EMG file types from various hardware/software combinations,
and convert them down to an array that can be further processed with helper_functions or other modules.
Additionally this file contains hash functions to track data
"""

import sys
import glob
import os

import pandas as pd
import numpy as np


import hashlib
import h5py
import re

# IMPORT Ole's fixed TMSiSDK python interfacwe- may change
sys.path.insert(0,'C:/Projects/tmsi-python-interface')
from TMSiSDK.file_readers import Poly5Reader

def poly5unpad(to_be_read):
    """
    This function converts a Poly5 read into an array without padding.
    Note there is a quirk in the python Poly5 interface that pads with zeros on the end.

    :param to_be_read: filename of python read Poly5
    :type to_be_read: :class:  str

    :return df: dataframe with hashes for what is in folder
    :rtype: :class: `~numpy.ndarray`
    """
    read_object=  Poly5Reader(to_be_read)
    sample_number= read_object.num_samples
    unpadded = read_object.samples[:, :sample_number]
    return unpadded


def hash_it_up_right_all(origin_folder1, file_extension):
    """
    Hashing function to check files are not corrupted or to assure files are changed.

    :param origin_folder1: the string of the folder with files to hash
    :type origin_folder1: :class:  str
    :param file_extension: file extension
    :type file_extension: :class: str

    :return df: dataframe with hashes for what is in folder
    :rtype: :class: `pandas.DataFrame`
    """
    hash_list = []
    file_names = []
    files = '*' + file_extension
    non_suspects1 = glob.glob(os.path.join(origin_folder1, files))
    BUF_SIZE = 65536
    for file in non_suspects1:
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        result = sha256.hexdigest()
        hash_list.append(result)
        file_names.append(file)
        
    df = pd.DataFrame(hash_list, file_names)
    df.columns = ["hash"]
    df = df.reset_index() 
    df = df.rename(columns = {'index':'file_name'})
    
    return df
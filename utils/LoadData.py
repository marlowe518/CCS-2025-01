#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from time import time
from sample import sampling
from get_privacy_spec import load_dict

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 500)


class DataSet:
    def __init__(self, path, privacy_spec=None, user_priv=None, random_seed=0):
        self.threshold = None
        self.train_by_users, self.train_by_items, \
        self.record_nums, self.user_nums, self.item_nums = self.load_grouped_rating_file_as_list(
            path + '_train_data', privacy_spec, user_priv, random_seed)
        self.test_ratings = self.load_rating_file_as_list(path + '_test_data')
        self.test_neg_ratings = self.load_neg_file_as_list(path + '_test_neg_data')

    def load_grouped_rating_file_as_list(self, filename, privacy_spec=None,
                                         user_priv=None, random_seed=None):  # load training data
        """Get two lists for users and items, e.g., [[uid, [iid0, iid1, ..., iidn]], ...]"""
        # Load the training
        data = self.load_rating_file_as_dataframe(filename)

        # Sampling mechanism
        if (privacy_spec is not None) and (user_priv is not None):
            threshold = 1
            data, threshold = sampling(data, privacy_spec, user_priv, threshold, random_seed=random_seed)
            self.threshold = threshold  # TODO 尝试优化一下这里的逻辑先后顺序，threshold的计算不应该放在dataset的读取里面

        # Get the nums of pos records, users, and items
        record_nums, user_nums, item_nums = data.shape[0], \
                                            data['uid'].max() + 1, data['iid'].max() + 1  # Nums: max_index + 1

        # Get grouped rating list for user and items
        train_by_users = self.get_grouped_rating_list(data, is_user=True)
        train_by_items = self.get_grouped_rating_list(data, is_user=False)
        self.train_by_users_or_items_distribution(train_by_items)
        return train_by_users, train_by_items, record_nums, user_nums, item_nums

    def train_by_users_or_items_distribution(self, arr):
        names = [str(tup[0]) for tup in arr]
        heights = [len(tup[1]) for tup in arr]
        import matplotlib.pyplot as plt
        plt.bar(np.array(names), np.array(heights))
        plt.show()
        return names, heights

    def load_rating_file_as_list(self, filename):  # load testing data
        """Get the testing list: [[uid, iid], ...]"""
        data = self.load_rating_file_as_dataframe(filename)
        rating_list = data.loc[:, ['uid', 'iid']].values.tolist()
        return rating_list

    def load_neg_file_as_list(self, filename):  # TODO be more general to cover the neg and test
        """Get the neg testing list: [[iid0, iid1, ..., iidn], ...]"""
        loop = True
        chunkSize = 5000
        chunks = []
        index = 0
        data = pd.read_table(filename, sep="\t", header=None, engine='python', iterator=True)
        # 分块读取数据
        while loop:
            try:
                chunk = data.get_chunk(chunkSize)
                chunks.append(chunk)
                index += 1
            except StopIteration:
                loop = False
        data = pd.concat(chunks, ignore_index=True)
        ratings = data.iloc[:, 1].apply(lambda x: list(map(int, x.split())))
        rating_list = ratings.values.tolist()  # retreive the neg samples
        return rating_list

    def load_rating_file_as_dataframe(self, filename):
        """Load training data as dataframe"""
        loop = True
        chunkSize = 5000
        chunks = []
        index = 0
        data = pd.read_table(filename, sep="\t", header=None, names=['uid', 'iid', 'rating'],
                             usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float}, engine='python',
                             iterator=True)
        # 分块读取数据
        while loop:
            try:
                chunk = data.get_chunk(chunkSize)
                chunks.append(chunk)
                index += 1
            except StopIteration:
                loop = False
        data = pd.concat(chunks, ignore_index=True)
        return data

    def get_grouped_rating_list(self, data: pd.DataFrame, is_user: bool):
        """Get the grouped ratings for users or items"""
        rating_list = []
        if is_user:
            user_groups = data.groupby('uid')
            for uid, group in user_groups:
                items = group['iid'].values
                rating_list.append((uid, list(items)))
            return rating_list
        else:
            item_groups = data.groupby('iid')
            for iid, group in item_groups:
                users = group['uid'].values
                rating_list.append((iid, list(users)))
        return rating_list

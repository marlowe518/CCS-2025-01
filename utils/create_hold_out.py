#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 500)


class MovieLens10MDataset:
    def __init__(self, DataPath, sep='::', header=None):
        self.raw_df, \
        self.user_nums, \
        self.item_nums = self._getDataset_as_list(DataPath, sep, header=header)

    def _getDataset_as_list(self, DataPath, sep, header):
        '''
        :param DataPath:
        :return: dataframe
        '''
        loop = True
        chunkSize = 5000
        chunks = []
        index = 0
        data = pd.read_table(DataPath, sep=sep, header=header, names=['uid', 'iid', 'rating', 'timestamp'],
                             usecols=[0, 1, 2, 3], dtype={0: np.int32, 1: np.int32, 2: np.float, 3: np.int32},
                             engine='python',
                             iterator=True)
        # 分块读取数据
        while loop:
            try:
                print('读取第{}块'.format(index))
                chunk = data.get_chunk(chunkSize)
                chunks.append(chunk)
                index += 1
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        print('开始合并')
        data = pd.concat(chunks, ignore_index=True)
        # 去除nan值
        nandata = data[np.isnan(data['rating'])].index
        data = data.drop(nandata)
        # 打印数据信息
        user_nums, item_nums = self.get_info(data)
        return data, user_nums, item_nums

    def get_info(self, df):
        '''
        介绍：用于打印数据基本信息
        :param df: dataframe
        :return:
        '''
        maxu, maxi, maxr = df['uid'].max() + 1, df['iid'].max() + 1, df['rating'].max()
        users, items = len(df['uid'].unique()), len(df['iid'].unique())
        sparseness = df.shape[0] / (users * items)
        print(
            'Data Statistics: Interaction = %d, maxuid = %d, maxiid = %d, Users = %d, items = %d, rating = %d, Sparsity = %.4f' % \
            (df.shape[0], maxu - 1, maxi - 1, users, items, maxr, sparseness))
        return users, items

    def rename(self, df):
        '''
        介绍：将用户和项目从0开始顺序编号
        :param df: dataframe columns=['uid','iid','rating]
        :return:
        '''
        uid_unique = df['uid'].unique()
        iid_unique = df['iid'].unique()
        temp = df['iid'].value_counts()
        user_rename_index = pd.DataFrame(np.vstack([uid_unique, np.arange(0, len(uid_unique))]).T,
                                         columns=['uid', 'newuid'], dtype='int')  # 从0开始编号
        item_rename_index = pd.DataFrame(np.vstack([iid_unique, np.arange(0, len(iid_unique))]).T,
                                         columns=['iid', 'newiid'], dtype='int')
        item_renames = pd.merge(df, item_rename_index, on='iid')  # 将新用户索引和评分数据合并
        all_renames = pd.merge(item_renames, user_rename_index, on='uid')
        renamed_df = all_renames[['newuid', 'newiid', 'rating', 'timestamp']].rename(
            columns={'newuid': 'uid', 'newiid': 'iid'})
        return renamed_df

    def rating_nums_truncate(self, df, userthresh, itemthresh):
        '''
        介绍：按照评分数量去除用户和项目
        :param data: dataframe
        :return:
        '''
        train_item_ratings = df['iid'].value_counts()
        print('项目评分数分箱分析：\n{}'.format(
            quicklook(train_item_ratings, [0, 20, 100, 500, 1000, 5000, np.inf])))
        print('平均每个项目评分数: {}'.format(train_item_ratings.mean()))
        train_user_ratings = df['uid'].value_counts()
        # print('用户评分数分箱分析：\n{}'.format(quicklook(train_user_ratings, [0, 20, 100, 500, 1000, 5000, np.inf])))
        # print('平均每个用户评分数: {}'.format(train_user_ratings.mean()))
        user_idxes = train_user_ratings[train_user_ratings >= userthresh].index
        item_idxes = train_item_ratings[train_item_ratings >= itemthresh].index
        truncated_df = df[df['uid'].isin(user_idxes)]
        truncated_df = truncated_df[truncated_df['iid'].isin(item_idxes)]
        return truncated_df

    def rating_nums_truncate_interval(self, df, userthresh, itemthresh_lower, itemthresh_upper):
        '''
        介绍：按照评分数量去除用户和项目
        :param data: dataframe
        :return:
        '''
        train_item_ratings = df['iid'].value_counts()
        print('项目评分数分箱分析：\n{}'.format(
            quicklook(train_item_ratings, [0, 20, 100, 500, 1000, 5000, np.inf])))
        print('平均每个项目评分数: {}'.format(train_item_ratings.mean()))
        train_user_ratings = df['uid'].value_counts()
        print('用户评分数分箱分析：\n{}'.format(quicklook(train_user_ratings, [0, 20, 100, 500, 1000, 5000, np.inf])))
        print('平均每个用户评分数: {}'.format(train_user_ratings.mean()))
        user_idxes = train_user_ratings[train_user_ratings >= userthresh].index
        item_idxes = train_item_ratings[train_item_ratings > itemthresh_lower][
            train_item_ratings <= itemthresh_upper].index
        print(len(item_idxes))
        truncated_df = df[df['uid'].isin(user_idxes)]
        truncated_df = truncated_df[truncated_df['iid'].isin(item_idxes)]
        return truncated_df

    def spilt_rating_data(self, data, size=0.2):
        '''
        介绍：按照比例划分训练集和测试集
        :param data: 2d ndarray
        :param size: split rate
        :return:
        '''
        train_data = []
        test_data = []
        for line in data:
            rand = np.random.random()
            if rand < size:
                test_data.append(line)
            else:
                train_data.append(line)
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        return train_data, test_data


def hold_out(raw_df: pd.DataFrame, item_nums, neg_sample_nums=99):
    user_groups = raw_df.groupby(raw_df['uid'])
    hold_out_idx = []
    neg_items = []
    for uid, group in user_groups:
        user_hold_out_idx = group['timestamp'].idxmax()
        hold_out_idx.append(user_hold_out_idx)

        # flush negs for user
        user_hold_out_item = group.loc[user_hold_out_idx, 'iid']  # get the hold-out item
        user_pos_items = group['iid'].values
        total_items = np.setdiff1d(np.arange(item_nums), user_pos_items)  # get all neg items
        user_neg_sampled = np.random.choice(total_items, size=neg_sample_nums, replace=False)  # sampling net items
        neg_items.append([(uid, user_hold_out_item), ' '.join(user_neg_sampled.astype(str))])

        # TODO shuffle user data ?

    test_data = raw_df.loc[hold_out_idx, :]
    train_data = raw_df.drop(index=hold_out_idx)
    test_neg_data = pd.DataFrame(neg_items, dtype='int')
    return test_data, train_data, test_neg_data


# 查看group的统计属性
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}


# 对data的数值进行分箱分析
def quicklook(data, bin_nums):
    '''
    :param data: 1d array
    :return:
    '''
    bins = pd.cut(data, bin_nums)
    grouped = pd.DataFrame(data).groupby(bins)
    look = grouped.apply(get_stats)  # 查看每一个bin的统计属性
    return look


def Kfold(dataset, k):
    '''
    介绍：按照K折划分数据集
    :param dataset:
    :param k:
    :return:
    '''
    dataset = np.array(dataset)
    kf = KFold(n_splits=k, random_state=1)
    n = 1
    for train, test in kf.split(dataset):
        print('trainset:{}\ntestset:{}\n'.format(train.shape, test.shape))
        trainset = pd.DataFrame(dataset[train], dtype='int')
        testset = pd.DataFrame(dataset[test], dtype='int')
        trainset.to_csv('./100k_trainset%d' % n, sep='\t', index=False, header=False)
        testset.to_csv('./100k_testset%d' % n, sep='\t', index=False, header=False)
        n += 1


def savefile(data, file):
    '''
    介绍：用于将dataframe存储为txt形式
    :param data: dataframe
    :param file:
    :return:
    '''
    # data['uid'] = data['uid'].astype('int')
    # data['iid'] = data['iid'].astype('int')
    # data['rating'] = data['rating'].astype('float')
    data.to_csv(file, sep='\t', header=False, index=False)
    print('保存成功！')


class MovieLens20MDataset(MovieLens10MDataset):
    def __init__(self, DataPath):
        super().__init__(DataPath, sep=',', header=0)


class MovieLens100KDataset(MovieLens10MDataset):
    def __init__(self, DataPath):
        super().__init__(DataPath, sep='\t', header=None)


class MovieLens25MDataset(MovieLens10MDataset):
    def __init__(self, DataPath):
        super().__init__(DataPath, sep=',', header=0)


class MovieLensYahooKDataset(MovieLens10MDataset):
    def __init__(self, DataPath):
        super().__init__(DataPath, sep='\t', header=None)

    def _getDataset_as_list(self, DataPath, sep, header):
        '''
        :param DataPath:
        :return: dataframe
        '''
        loop = True
        chunkSize = 5000
        chunks = []
        index = 0
        data = pd.read_table(DataPath, sep=sep, header=header, names=['uid', 'iid', 'rating'],
                             usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float},
                             engine='python',
                             iterator=True)
        # 分块读取数据
        while loop:
            try:
                print('读取第{}块'.format(index))
                chunk = data.get_chunk(chunkSize)
                chunks.append(chunk)
                index += 1
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        print('开始合并')
        data = pd.concat(chunks, ignore_index=True)
        # 去除nan值
        nandata = data[np.isnan(data['rating'])].index
        data = data.drop(nandata)
        # add timestamp (all=1) to yahoomusic dataset
        data['timestamp'] = np.ones(data.shape[0])
        # 打印数据信息
        user_nums, item_nums = self.get_info(data)
        return data, user_nums, item_nums


class InstantVideoDataset(MovieLens10MDataset):
    def __init__(self, DataPath):
        super().__init__(DataPath, sep=',', header=None)

    def _getDataset_as_list(self, DataPath, sep, header):
        data = pd.read_table(DataPath, sep=sep, header=header, names=['uid', 'iid', 'rating'],
                             usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float})
        # add timestamp (all=1) to yahoomusic dataset
        data['timestamp'] = np.ones(data.shape[0])
        # 打印数据信息
        user_nums, item_nums = self.get_info(data)
        return data, user_nums, item_nums


def batch_processing(datatype='ml-10m'):
    if datatype == 'ml-10m':
        datafile = '../data/ml-10M100K/ratings.dat'
        DS = MovieLens10MDataset(datafile)
    elif datatype == 'ml-20m':
        datafile = '../data/ml-20m/ratings.csv'
        DS = MovieLens20MDataset(datafile)
    elif datatype == 'ml-25m':
        datafile = '../data/ml-25m/ratings.csv'
        DS = MovieLens25MDataset(datafile)
    elif datatype == 'ml-100k':
        datafile = '../data/ml-100k/u.data'
        DS = MovieLens100KDataset(datafile)
    elif datatype == 'ml-1m':
        datafile = '../data/ml-1m/ratings.dat'
        DS = MovieLens10MDataset(datafile)
    elif datatype == 'yahoo_music':
        datafile = '../data/yahoomusic/dataset'
        DS = MovieLensYahooKDataset(datafile)
    elif datatype == 'amazon':
        datafile = '../data/IV/dataset(IV).csv'
        DS = InstantVideoDataset(datafile)
    else:
        raise ValueError('unknown data name: ' + datatype)
    DS.rating_nums_truncate_interval(DS.raw_df, 0, 0, 0)
    renamed_df = DS.rename(DS.raw_df)
    test_data, train_data, test_neg_data = hold_out(renamed_df, item_nums=DS.item_nums)
    savefile(test_data, datatype + "_test_data")
    savefile(train_data, datatype + "_train_data")
    savefile(test_neg_data, datatype + "_test_neg_data")


if __name__ == '__main__':
    # 批处理
    datatype = 'yahoo_music'
    # datatype = 'amazon'
    batch_processing(datatype=datatype)

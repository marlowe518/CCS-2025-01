
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time

# from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [], []
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating_checked, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):  # evaluate each user, then average
        (hr, ndcg) = eval_one_rating_checked(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


# def eval_one_rating(idx):
#     rating = _testRatings[idx]
#     items = _testNegatives[idx] # the sampled neg_items in test
#     u = rating[0]
#     gtItem = rating[1] # the pos item in test
#     items.append(gtItem)
#     # Get prediction scores
#     map_item_score = {}
#     users = np.full(len(items), u, dtype='int32') # fill the item_num-length array with uid
#     predictions = _model.predict([users, np.array(items)], # input 2 item_num-length array
#                                  batch_size=100, verbose=0)
#     for i in range(len(items)):
#         item = items[i]
#         map_item_score[item] = predictions[i]
#     items.pop() # the gtItem is removed TODO when items will not be used anymore?
#
#     # Evaluate top rank list
#     ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
#     hr = getHitRatio(ranklist, gtItem)
#     ndcg = getNDCG(ranklist, gtItem)
#     return (hr, ndcg)

def eval_one_rating_checked(idx):  # Fix that test rating may include the item doesn't exist in train data.
    rating = _testRatings[idx]
    items = _testNegatives[idx]  # the sampled neg_items in test
    u = rating[0]
    gtItem = rating[1]  # the pos item in test
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')  # fill the item_num-length array with uid
    predictions = _model.predict([users, np.array(items)],  # input 2 item_num-length array
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()  # the gtItem is removed TODO when items will not be used anymore?

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

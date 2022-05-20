#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/13
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: R_similarity_distribution_analysis.py
import argparse
import json
import os

import numpy as np
import scipy.spatial
from tqdm import tqdm

SUBSET = ['val', 'train', 'test']


def vector_distance(vec1, vec2, method = "l2", l2_normalize = True):
    """Computes the distance between 2 vectors
    Inspired by https://github.com/Azure/ImageSimilarityUsingCntk=
    Args:
        vec1: First vector between which the distance will be computed
        vec2: Second vector
        method: Type of distance to be computed, e.g. "l1" or "l2"
        l2_normalize: Flag indicating whether the vectors should be normalized
        to be of unit length before the distance between them is computed
    Returns: Distance between the 2 input vectors
    """
    # Pre-processing
    if l2_normalize:
        vec1 = vec1 / np.linalg.norm(vec1, 2)
        vec2 = vec2 / np.linalg.norm(vec2, 2)

    # Distance computation
    vecDiff = vec1 - vec2
    method = method.lower()
    if method == "l1":
        dist = sum(abs(vecDiff))
    elif method == "l2":
        dist = np.linalg.norm(vecDiff, 2)
    elif method == "normalizedl2":
        a = vec1 / np.linalg.norm(vec1, 2)
        b = vec2 / np.linalg.norm(vec2, 2)
        dist = np.linalg.norm(a - b, 2)
    elif method == "cosine":
        dist = scipy.spatial.distance.cosine(vec1, vec2)
    elif method == "correlation":
        dist = scipy.spatial.distance.correlation(vec1, vec2)
    elif method == "chisquared":
        dist = scipy.chiSquared(vec1, vec2)
    elif method == "normalizedchisquared":
        a = vec1 / sum(vec1)
        b = vec2 / sum(vec2)
        dist = scipy.chiSquared(a, b)
    elif method == "hamming":
        dist = scipy.spatial.distance.hamming(vec1 > 0, vec2 > 0)
    else:
        raise Exception("Distance method unknown: " + method)
    return dist

def read_feat(args, info):
    "00005197-869d72f3-66210bf4-fa2c9d83-b613c4e7.npy, (49, 1024)"
    "00005197-869d72f3-66210bf4-fa2c9d83-b613c4e7_tag.npy, (14)"
    path = os.path.split(info['path'])[-1]
    path = os.path.splitext(path)[0] + '.npy'
    path = os.path.join(args.featdir, path)

    vec = np.load(path)
    vec = np.mean(vec, axis=1)
    vec_norm = np.linalg.norm(vec, 2)
    return vec / vec_norm



def main(args, topk=10):
    with open(args.json, 'r') as f:
        data = json.load(f)
    """
    {'train': {items}, 'val': {items}, 'test': {items}}
    item:
    "19271682": [{"study_id": 58170907, "report": "", "path": "files/p19/p19271682/s58170907/9d2862f4-a895e619-d04d3f23-6a68a4d3-faf3f00b.dcm", "date": 21900210}, 
                 {"study_id": 56445853, "report": "", "path": "files/p19/p19271682/s56445853/6ab3598c-cea4d4f2-ef883b5b-c77ed966-cfb5dcd4.dcm", "date": 21901128}, 
                 {"study_id": 56687967, "report": "", "path": "files/p19/p19271682/s56687967/ca178126-75360a21-cebfc3a1-29551687-d1317dec.dcm", "date": 21910217}, 
                 {"study_id": 54706003, "report": "", "path": "files/p19/p19271682/s54706003/afa18b4a-5d8cb5c0-a4d9cf54-d3339089-2142462a.dcm", "date": 21910219}, 
                 {"study_id": 54387416, "report": "", "path": "files/p19/p19271682/s54387416/e05630e8-462adcc6-6b1109ac-53b6181b-889ca848.dcm", "date": 21910325}, 
                 {"study_id": 55658177, "report": "", "path": "files/p19/p19271682/s55658177/ee3f40bc-e7599d92-7be286c1-e35e7e75-7714c4a1.dcm", "date": 21910416}
                 ]
    """
    # 设置随机种子
    np.random.seed(123)

    # 找出所有病历数量大于一次的病人
    feats = []
    for k, item in tqdm(data['train'].items(), ncols=80):
        if(len(item)) > 1:
            feat = []
            feat.append(read_feat(args, item[0]))
            feat.append(read_feat(args, item[1]))

            feats.append(feat)

    # 保存平均 Rank
    ranks = []
    top1, top3, top5 = 0, 0, 0

    # 为这些病人随机匹配9个其他病人的病历
    # 通过cosine similarity 计算同一个病人的图片在10张图片中的排名
    index = [i for i in range(len(feats))]
    for i, item in enumerate(tqdm(feats, ncols=80)):
        # 用作相似度比较的图像
        target = item[1]

        # 病人的自身的前一次图像
        same_before = item[0]

        # 计算similarity score
        same_simi_score = np.dot(target, same_before)

        # sample的9张其他病人的图像
        others_id = np.random.choice(index, 9)

        # 计算另外9张的similarity score
        other_simi_scores = []
        for j in others_id:
            if j == i :
                j = j + 1 if j + 1 < len(feats) else j - 1
            simi_score = np.dot(target, feats[j][0])
            other_simi_scores.append(simi_score)

        # 因为只有10个数，可以直接排序后顺序查找
        other_simi_scores = sorted(other_simi_scores, reverse=True)
        r = 1
        for val in other_simi_scores:
            if same_simi_score < val:
                r += 1
        if r == 1: top1 += 1
        if r <= 3: top3 += 1
        if r <= 5: top5 += 1
        ranks.append(r)

    mean_rank = np.mean(ranks)
    print("Mean Rank: ", mean_rank)
    print("Top@1:", top1 / len(ranks))
    print("Top@3:", top3 / len(ranks))
    print("Top@5:", top5 / len(ranks))

    # 计算同一个病人的图片和不同病人图片之间分布的KL散度



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', type=str, default="/home/shuxinyang/data/mimic/finding/pair_list.json",
                        help="data information")
    parser.add_argument('-f', '--featdir', type=str, default="/home/shuxinyang/data/mimic/features",
                        help="feature dir")
    parser.add_argument('-o', '--output', type=str, default="/home/shuxinyang/data/mimic/finding/",
                        help="output file save dir")
    parser.add_argument('-m', '--method', type=str, default="cosine",
                        help="similarity method")

    args = parser.parse_args()
    print(args)

    main(args)
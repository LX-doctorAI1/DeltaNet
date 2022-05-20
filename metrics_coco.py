import sys

sys.path.append("..")

import argparse
import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from metrics import cal_epoch as cal_epoch_old


# METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDEr', 'ROUGE_L', 'METEOR']
METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDEr', 'ROUGE_L']


def compute_scores(gts, res):
    """
    Adapted from https://github.com/tylin/coco-caption
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions, {id: [caption1, caption2]}
    :param res: Dictionary with the image ids ant their generated captions, {id: [caption1, caption2]}
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    eval_res = {}
    eval_res['METEOR'] = 0.0

    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def cal_epochs(args, epochs, mode='val'):
    '''读取所有生成的报告文件用于评测'''
    # filename format: "Enc2Dec-{epoch}_{mode}_generated.json"
    file_list = glob(args.expe_name + f'*{mode}*.json')
    files = dict()
    for fl in file_list:
        e = int(os.path.split(fl)[1].split('_')[0].split('-')[1])
        files[e] = fl

    epochs = [e for e in epochs if e in files.keys()]
    res = []
    try:
        for epoch in tqdm(epochs):
            if args.old:
                result = cal_epoch_old(files[epoch])
            else:
                result = cal_epoch(files[epoch])
            result['epoch'] = epoch
            res.append(result)
    except Exception as e:
        print('===>> Metric Stop : {} '.format(e))
    return res


def cal_epoch(filename):
    '''读取生成的报告json文件，转换成dict后计算'''
    with open(filename, 'r') as f:
        data = json.load(f)

    gts, res = dict(), dict()
    for i, item in enumerate(data):
        pred = item['prediction']
        gt = item['ground_truth']

        # Covid数据集未分割时
        # pred = ' '.join([c for c in pred])
        # gt = ' '.join([c for c in gt])

        gts[i] = [gt]
        res[i] = [pred]

    eval_res = compute_scores(gts, res)
    return eval_res


def merge_res(done_res, val_res, test_res):
    '''合并结果'''
    for val, test in zip(val_res, test_res):
        assert val['epoch'] == test['epoch'], 'result order must be same'
        item = {'EPOCH': val['epoch']}
        for m in METRICS:
            item[f'VAL_{m}'] = val[m]
            item[f'TEST_{m}'] = test[m]
        done_res = done_res.append(item, ignore_index=True)
    done_res.sort_values(by=['VAL_BLEU_4'], ascending=False, inplace=True)
    return done_res


def main(args):
    # 保存结果的文件名
    expe_path = os.path.split(os.path.normpath(args.expe_name))[0]
    expe_prefix = os.path.basename(expe_path)
    if args.old:
        result_file = os.path.join(args.expe_name, expe_prefix + '_result_old.csv')
    else:
        result_file = os.path.join(args.expe_name, expe_prefix + '_result.csv')
    # 读取之前做过的评测结果，避免重复计算
    if os.path.exists(result_file):
        done_res = pd.read_csv(result_file)
        done_epoch = done_res['EPOCH']
    else:
        done_res = pd.DataFrame()
        done_epoch = []

    epochs = [e for e in range(args.start, int(args.epoch) + 1) if e not in done_epoch]

    # select num of best (args.nbest) cider epoch
    if not args.test:
        # got best N val epoch by monitor method
        val_res = cal_epochs(args, epochs, 'val')
        # got test result by best N val epoch
        test_res = cal_epochs(args, epochs, 'test')

        # 将val和test上的结果合并成一行，加到csv文件最后
        res = merge_res(done_res, val_res, test_res)
        res.to_csv(result_file, index=False)

        print('*' * 20 + f'VAL SET BEST {args.nbest}' + '*' * 20)
        print(res.sort_values(f'VAL_{args.method}', ascending=False).head(args.nbest))
        print('*' * 20 + f'TEST SET BEST {args.nbest}' + '*' * 20)
        print(res.sort_values(f'TEST_{args.method}', ascending=False).head(args.nbest))

    else:
        # got best N test epoch by monitor method
        test_res = cal_epochs(args, epochs, 'test')
        res = merge_res(done_res, test_res, test_res)

        print('*' * 20 + f'TEST SET BEST {args.nbest}' + '*' * 20)
        print(res.sort_values(f'TEST_{args.method}', ascending=False).head(args.nbest))

    print(args.root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('-n', '--expe_name', type=str, default='')
    parser.add_argument('-m', '--method', type=str, default='CIDEr', choices=['CIDEr', 'BLEU_4'])

    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('--old', action='store_true')

    parser.add_argument('-e', '--epoch', type=str, default=100)
    parser.add_argument('-s', '--start', type=int, default=1)
    parser.add_argument('-nb', '--nbest', type=int, default=5)

    args = parser.parse_args()
    assert args.root != '' or args.expe_name != '', "files path don't set"
    args.expe_name = os.path.join(args.root, args.expe_name)
    print(args)

    main(args)

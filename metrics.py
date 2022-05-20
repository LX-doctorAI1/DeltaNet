import sys
sys.path.append("..")
import argparse
import json
import os

from pycocoevalcap.eval import calculate_metrics


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def cal_epoch(result_path, epoch=0, mode='all', covid=False):
    test = load_json(result_path)
    datasetGTS = {'annotations': []}
    datasetRES = {'annotations': []}

    for i, data in enumerate(test):
        if covid:
            pred_sent = ' '.join([c for c in data["prediction"]])
            real_sent = ' '.join([c for c in data["ground_truth"]])
        else:
            pred_sent = data["prediction"]
            real_sent = data["ground_truth"]
        datasetGTS['annotations'].append({
            'image_id': i,
            'caption': real_sent
        })
        datasetRES['annotations'].append({
            'image_id': i,
            'caption': pred_sent
        })

    rng = range(len(test))
    res = calculate_metrics(rng, datasetGTS, datasetRES, mode)
    res['epoch'] = epoch
    # print(res)
    return res


def cal_epochs(args, epochs, subset='val', mode='all', covid=False):
    bleu_res = []
    try:
        for epoch in epochs:
            print('*' * 20 + str(epoch) + '*' * 20)
            result_path = os.path.join(args.expe_name, "Enc2Dec-" + str(epoch) + "_{}_generated.json".format(subset))
            result = cal_epoch(result_path, epoch, mode, covid)
            bleu_res.append(result)
    except Exception as e:
        print('===>> Metric Stop : {} '.format(e))
    return bleu_res


def print_res(metric_res, metric_list):
    for i in range(len(metric_res)):
        str_res = ','.join([str(metric_res[i][m]) for m in metric_list])
        print(str_res)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=' /home/shuxinyang/data/iu/r2gen_split/checkpoints')
    parser.add_argument('-n', '--expe_name', type=str, default='')
    parser.add_argument('-m', '--method', type=str, default='CIDEr', choices=['CIDEr', 'Bleu_4'])
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-c', '--covid', action='store_true')

    parser.add_argument('-e', '--epoch', type=str, default=100)
    parser.add_argument('-s', '--start', type=int, default=1)
    parser.add_argument('-nb', '--nbest', type=int, default=5)

    args = parser.parse_args()

    metric_list = ['epoch', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr', 'ROUGE_L', 'METEOR']
    # metric_list = ['epoch', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr', 'ROUGE_L']
    head = '\n' + '*' * 20 + " Summery " + '*' * 20 + '\n' + ','.join(metric_list)
    args.expe_name = os.path.join(args.root, args.expe_name)

    metric_res, bleu_res, test_metric_res = [], [], []
    # select num of best (args.nbest) cider epoch
    if not args.test:
        # got best N val epoch by cider
        epochs = range(args.start, int(args.epoch) + 1)
        bleu_res = cal_epochs(args, epochs, 'val', 'cider', args.covid)

        bleu_res = sorted(bleu_res, key=lambda x: x[args.method], reverse=True)

        nbest = args.nbest if args.nbest < len(bleu_res) else len(bleu_res)
        best_epochs = [bleu_res[i]['epoch'] for i in range(nbest)]

        # got best N val epoch all metric
        # metric_res = cal_epochs(args, best_epochs, 'val', 'all', args.covid)

        # got test result by best N val epoch
        test_metric_res = cal_epochs(args, best_epochs, 'test', 'all', args.covid)

        # output result
        print(head)
        print_res(bleu_res[:nbest], metric_list)
        # print_res(metric_res, metric_list)
        print_res(test_metric_res, metric_list)

    else:
        # got best N test epoch by bleu
        epochs = range(args.start, int(args.epoch) + 1)
        bleu_res = cal_epochs(args, epochs, 'test', 'bleu', args.covid)
        bleu_res = sorted(bleu_res, key=lambda x: x[args.method], reverse=True)

        nbest = args.nbest if args.nbest < len(bleu_res) else len(bleu_res)
        best_epochs = [bleu_res[i]['epoch'] for i in range(args.nbest)]

        # got best N val epoch all metric
        test_metric_res = cal_epochs(args, best_epochs, 'test', 'all', args.covid)

        print(head)
        print_res(test_metric_res, metric_list)

    print(args.root)

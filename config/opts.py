from __future__ import print_function

import argparse
import json
import os
from datetime import datetime, timedelta


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument('--root', type=str, default='/home/shuxinyang/data/mimic/finding/',
                        help='root path for experiment')
    parser.add_argument('--expe_name', type=str, default='con1_similar_sup',
                        help='path for saving trained models')
    parser.add_argument('--N', type=int, default=1,
                        help='number of similar')
    parser.add_argument('--basic_model', type=str, default='VisualAttention',
                        help='the selected basic model')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='',
                        help='directory for training images')
    parser.add_argument('--caption_json', type=str, default='',
                        help='path for train annotation json file')
    parser.add_argument('--pair_list', type=str, default='',
                        help='path for image concepts json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')

    # ---------------------------Hyper Parameter Setup------------------------------------
    # CNN fine-tuning
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=8,
                        help='start fine-tuning CNN after')
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-5,
                        help='learning rate for fine-tuning CNN')

    # Optimizer Adam parameter
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='alpha in Adam')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='beta in Adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate for the whole model')

    # LSTM hyper parameters
    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')

    # Training details
    parser.add_argument('--pretrained', type=str, default='', help='start from checkpoint or scratch')
    parser.add_argument('--pretrained_cnn', type=str, default='', help='load pertraind_cnn parameters')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--lr_decay', type=int, default=10, help='epoch at which to start lr decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=10,
                        help='decay learning rate at every this number')

    # Generattion parameters
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size')
    parser.add_argument('--max_length', type=int, default=60, help='The maximum length of generated target.')
    parser.add_argument('--n_best', type=int, default=1, help="""If verbose is set, will output the n_best
                                    decoded sentences""")

    parser.add_argument('--dataset', type=str, default='mimic', help='dataset name')
    parser.add_argument('-g', '--gpu', type=str, default="4")
    parser.add_argument('-s', '--start_gen', type=int, default=1)

    # config
    parser.add_argument('--cfg', type=str, default=None,
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from .config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    # Check if args are valid
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("=====>> Set GPU devices: ", args.gpu)

    # add training parameters to save dir
    expe_name = args.expe_name + (datetime.now() + timedelta(hours=15)).strftime("_%Y%m%d-%H%M%S")
    args.expe_name = os.path.join(args.root, 'checkpoints', expe_name)

    # Save config for reproduce
    if not os.path.exists(args.expe_name):
        os.makedirs(args.expe_name)
    with open(os.path.join(args.expe_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    return args


def add_eval_options(parser):
    pass


def add_diversity_opts(parser):
    pass


# Sampling related options
def add_eval_sample_opts(parser):
    pass


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0]]
    args = parse_opt()
    print(args)
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml']
    args1 = parse_opt()
    print(dict(set(vars(args1).items()) - set(vars(args).items())))
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml', '--visual_extractor', 'densenet']
    args2 = parse_opt()
    print(dict(set(vars(args2).items()) - set(vars(args1).items())))
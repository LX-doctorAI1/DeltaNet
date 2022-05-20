import json
import os
import pickle
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from Models.Generator import Generator
from Models.MultiConditional import Encoder2Decoder
from Models.misc import idx2words, to_var, cal_loss, get_transform, seed_everything, adjust_learning_rate
from data_loader import get_loader
from metrics_coco import cal_epoch, merge_res
from config import opts


METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDEr', 'ROUGE_L']


def get_data_loader(args, vocab, mode='train'):
    # print('=======>> Load {} dataset <<=========='.format(mode))
    batch_size = args.batch_size
    transform = get_transform(args, mode)
    data_loader = get_loader(args.image_dir, args.caption_json, args.pair_list, vocab,
                             transform, batch_size, shuffle=(mode == 'train'), max_len=args.max_length,
                             num_workers=args.num_workers, type=args.dataset, N=args.N, subset=mode)

    return data_loader


def train(args, Model, optimizer, data_loader, epoch, cnn_optimizer=None):
    Model.train()
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch}/{args.num_epochs}', unit='batch') as pbar:
        for i, (images, target, lengths, _, _, prev_repo) in enumerate(data_loader):
            # Set mini-batch dataset
            images, target, prev_repo = to_var(images), to_var(target), to_var(prev_repo)
            lengths = [cap_len - 1 for cap_len in lengths]
            targets = pack_padded_sequence(target[:, 1:], lengths, batch_first=True)[0]
            optimizer.zero_grad()
            cnn_optimizer.zero_grad()
            # Forward, Backward and Optimize
            packed_scores = Model(images, target, prev_repo, lengths, args.basic_model)
            # Compute loss and backprop
            loss = cal_loss(packed_scores[0], targets, smoothing=True)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update(1)

            # optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for gradient exploding problem in LSTM
            nn.utils.clip_grad_norm(Model.parameters(), args.clip)
            optimizer.step()

            if epoch > args.cnn_epoch:
                cnn_optimizer.step()

    # Save the Model after each epoch
    # Create model directory
    save_path = os.path.join(args.expe_name, args.basic_model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the Model
    torch.save(Model.state_dict(), os.path.join(save_path, 'Enc2Dec-%d.pkl' % (epoch)))
    return Model


def validation(args, model, data_loader, vocab, epoch, mode='val'):
    Caption_Generator = Generator(args, model)
    results = []
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch}/{args.num_epochs}', unit='batch') as pbar:
        for i, (images, target, lengths, _, image_ids, prev_repo) in enumerate(data_loader):
            images = to_var(images)
            prev_repo = to_var(prev_repo)
            all_hyp, all_scores = Caption_Generator.translate_batch(images, prev_repo)

            # Build caption based on Vocabulary and the '<end>' token
            batch_bleu = list()
            prefix = ' '
            prev_repo = prev_repo.cpu().numpy()
            for image_idx in range(len(all_hyp)):
                sampled_caption = idx2words(all_hyp[image_idx][0], vocab)
                best_sentence = ' '.join(sampled_caption)
                gt_caption = idx2words(target[image_idx].numpy()[1:], vocab)
                ground_truth = ' '.join(gt_caption)

                bleu4 = sentence_bleu([gt_caption], sampled_caption)
                temp = {'image_id': image_ids[image_idx], 'prediction': best_sentence,
                        'ground_truth': ground_truth, 'BLEU4': bleu4}
                for j in range(args.N):
                    prev_caption = idx2words(prev_repo[image_idx][j][1:], vocab)
                    prev_sentence = prefix.join(prev_caption)
                    temp['previous{}'.format(j + 1)] = prev_sentence
                results.append(temp)
                batch_bleu.append(bleu4)

            pbar.set_postfix(**{'BLEU-4 (batch)': np.asarray(batch_bleu).mean()})
            pbar.update(1)

    save_path = os.path.join(args.expe_name, args.basic_model)
    results = sorted(results, key=lambda x: x['BLEU4'], reverse=True)
    resFile = os.path.join(save_path, 'Enc2Dec-{}_{}_generated.json'.format(epoch, mode))
    json.dump(results, open(resFile, 'w'))
    print('===>> {}/Enc2Dec-{}_{} Generated'.format(save_path, epoch, mode))

    # output bleu result
    eval_res = cal_epoch(resFile)
    eval_res['epoch'] = epoch
    print(f'Epoch {epoch}||| ' + ' |||'.join(['{}: {:.4}'.format(m, eval_res[m]) for m in METRICS]))
    return eval_res


# Main Function
def main(args):
    # To reproduce training results
    seed_everything(args.seed)

    # 保存评测结果
    expe_path = os.path.split(os.path.normpath(args.expe_name))[0]
    expe_prefix = os.path.basename(expe_path)
    result_file = os.path.join(args.expe_name, args.basic_model, expe_prefix + '_result.csv')
    if os.path.exists(result_file):
        done_res = pd.read_csv(result_file)
    else:
        done_res = pd.DataFrame()

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load pretrained model or build from scratch
    model = Encoder2Decoder(args.embed_size, len(vocab), args.hidden_size, args.N)
    # Change to GPU mode if available
    if torch.cuda.is_available():
        model.cuda()

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int(args.pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1

    elif args.pretrained_cnn:
        pretrained_dict = torch.load(args.pretrained_cnn)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        start_epoch = 1

    else:
        start_epoch = 1

    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_params = list(model.encoder_image.parameters())

    # Parameter optimization
    params = list(model.encoder_concept.parameters()) \
             + list(model.decoder.parameters())

    # Will decay later
    lr, cnn_lr = args.learning_rate, args.learning_rate_cnn
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=cnn_lr, betas=(args.alpha, args.beta))
    optimizer = torch.optim.Adam(params, lr=lr, betas=(args.alpha, args.beta))

    # Build training data loader
    train_data_loader = get_data_loader(args, vocab, 'train')
    val_data_loader = get_data_loader(args, vocab, 'val')
    test_data_loader = get_data_loader(args, vocab, 'test')

    # Train the models
    # Start Training
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            adjust_learning_rate(optimizer, lr, args, epoch, 'Model')
            adjust_learning_rate(cnn_optimizer, cnn_lr, args, epoch, 'CNN')

            # Language Modeling Training
            print('\n------------------Training for Epoch %d----------------' % (epoch))
            model.decoder.condition.gates = None
            model = train(args, model, optimizer, train_data_loader, epoch, cnn_optimizer)
            gates = model.decoder.condition.gates
            print(torch.mean(torch.nn.functional.softmax(gates), dim=0))
            if epoch >= args.start_gen:
                print('------------------Validating for Epoch %d----------------' % (epoch))
                val_res = validation(args, model, val_data_loader, vocab, epoch, 'val')

                print('------------------Testing for Epoch %d----------------' % (epoch))
                test_res = validation(args, model, test_data_loader, vocab, epoch, 'test')

                done_res = merge_res(done_res, [val_res], [test_res])
    except KeyboardInterrupt:
        print('=====>> Early Stop!')
    finally:
        if not done_res.empty:
            done_res.to_csv(result_file, index=False)


if __name__ == '__main__':
    args = opts.parse_opt()

    print('------------------------Model and Training Details--------------------------')
    print(args)

    # Start training
    main(args)


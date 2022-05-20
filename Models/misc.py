import json
import math
import os
import pickle
import random

import bcolz
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import Models.Constants as Constants


# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def idx2words(sampled_ids, vocab):
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        else:
            sampled_caption.append(word)
    return sampled_caption


def save_res(results, resFile):
    with open(resFile, 'w') as f:
        json.dump(results, f, ensure_ascii=False)


def save_res_nopre(results, resFile):
    with open(resFile, 'w') as f:
        json.dump(results, f, ensure_ascii=False)


def sentence_postprec(prefix, sample):
    result = prefix.join(sample)
    # result = result.replace(' ,', ',').replace(' .', '.')
    return result


# Create tgt_pos
def generate_tgt_pos(batch_size, length):
    tgt_pos = [i + 1 for i in range(max(length) + 1)]
    tgt_pos = np.array(tgt_pos)
    tgt_pos = Variable(torch.from_numpy(tgt_pos))
    tgt_pos = tgt_pos.unsqueeze(0).expand(batch_size, -1)
    Tgt = torch.zeros(batch_size, max(length) + 1).long()
    for i in range(batch_size):
        end = length[i]
        Tgt[i, :end] = tgt_pos[i, :end]
    return to_var(Tgt)


def get_glove(args, target_vocab):
    assert args.embed_size == 300, "glove embed size is 300!"

    vectors = bcolz.open(os.path.join(args.glove_path, '6B.300.dat'))[:]
    words = pickle.load(open(os.path.join(args.glove_path, '6B.300_words.pkl'), 'rb'))
    word2idx = pickle.load(open(os.path.join(args.glove_path, '6B.300_idx.pkl'), 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for word in target_vocab.word2idx.keys():
        i = target_vocab(word)
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300,))

    return weights_matrix


def cal_loss(pred, target, smoothing=True):
    # ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    target = target.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = target.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        # loss = loss.masked_select(non_pad_mask).sum()
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # Language Modeling Loss
        LMcriterion = nn.CrossEntropyLoss()
        # Change to GPU mode if available
        if torch.cuda.is_available():
            LMcriterion.cuda()
        loss = LMcriterion(pred, target)

    return loss

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def vector_distance(vec1, vec2, method="l2", l2_normalize=True):
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
    eps = 1e-6
    # Pre-processing
    if l2_normalize:
        vec1 = vec1 / (np.linalg.norm(vec1, 2) + eps)
        vec2 = vec2 / (np.linalg.norm(vec2, 2) + eps)

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


def get_transform(args, mode='train'):
    if mode == 'train':
        # Image Preprocessing
        # For normalization, see https://github.com/pytorch/vision#models
        transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            # transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    return transform


def adjust_learning_rate(optimizer, lr, args, epoch, prefix=''):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch > args.lr_decay:
        frac = float(epoch - args.lr_decay) / args.learning_rate_decay_every
        decay_factor = math.pow(0.5, frac)
        # Decay the learning rate
        lr *= decay_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(prefix + ' Learning Rate for Epoch %d: %.6f' % (epoch, lr))

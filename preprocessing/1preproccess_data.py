import sys
sys.path.append('..')
import argparse
import json
import os
import pickle
import pandas as pd
from collections import Counter
import os.path as osp
from sklearn.model_selection import train_test_split

from Models.utils import normalize_text
from build_vocab import Vocabulary

SUBSET = ['train', 'val', 'test']


def build_vocab(report_data, save_dir, threshold=2):
    """Build a simple vocabulary wrapper."""
    counter = Counter()

    for i, idx in enumerate(report_data):
        tokens = report_data[idx].split()
        counter.update(tokens)

        if i % 500 == 0:
            print("[%d/%d] Tokenized the captions." % (i, len(report_data)))

    # print(counter)
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # dep = [word for word, cnt in counter.items() if cnt < threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    vocab_path = osp.join(save_dir, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" % len(vocab))
    print("Saved the vocabulary wrapper to '%s'" % vocab_path)
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root', type=str, default="data",
                        help="the begin directory of all")
    parser.add_argument('-r', '--report', type=str, default="indiana_reports.csv",
                        help="IU report file")
    parser.add_argument('-p', '--projections', type=str, default="indiana_projections.csv",
                        help="indiana projections file")
    parser.add_argument('-o', '--output', type=str, default="",
                        help="output file save dir")

    args = parser.parse_args()
    args.projections = os.path.join(args.root, args.projections)
    args.report = os.path.join(args.root, args.report)
    args.output = os.path.join(args.root, args.output)
    print(args)

    # drop nan finding and get frontal filenames
    proj = pd.read_csv(args.projections)
    report = pd.read_csv(args.report)

    iu_data = pd.merge(proj, report)
    iu_data = iu_data.dropna(subset=["findings"])
    iu_data = iu_data[iu_data['projection'] == 'Frontal']

    # split to train: ～0.7, val: ～0.1, test: 0.2
    # first split in uid, then get filename(someone has two frontal file)
    train_val_id, test_id = train_test_split(list(set(iu_data['uid'])), test_size=0.2)
    train_id, val_id = train_test_split(train_val_id, test_size=0.12)

    iu_filename = iu_data['filename']
    train_filename = iu_filename[iu_data['uid'].isin(list(train_id))]
    val_filename = iu_filename[iu_data['uid'].isin(list(val_id))]
    test_filename = iu_filename[iu_data['uid'].isin(list(test_id))]

    if not osp.exists(args.output):
        os.makedirs(args.output)
    train_filename.to_csv(osp.join(args.output, 'train_split.csv'), index=False, header=True)
    val_filename.to_csv(osp.join(args.output, 'val_split.csv'), index=False, header=True)
    test_filename.to_csv(osp.join(args.output, 'test_split.csv'), index=False, header=True)

    # save finding to json
    caption = {}
    for uid, finding in zip(iu_data['uid'], iu_data['findings']):
        clean_finding = normalize_text(finding)
        caption[str(uid)] = clean_finding

    with open(osp.join(args.output, 'idx2caption.json'), 'w') as f:
        json.dump(caption, f)

    # build and save vocabulary
    build_vocab(caption, args.output, threshold=3)

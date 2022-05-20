import json
import os

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image


class _DatasetBase(data.Dataset):
    """自定义dagaset的基类
    提供公用init方法和抽象函数
    """
    def __init__(self, image_dir, json_report, pair_split, vocab, max_len=80, N=1, subset='train', transform=None):
        self.image_dir = image_dir
        self.json_report = json_report
        self.pair_split = pair_split
        self.vocab = vocab
        self.max_len = max_len
        self.N = N
        self.subset = subset
        self.transform = transform

        """
        report 文件:
        IU:
            (key: 病人编号, value: 病人编号)
            {病人编号: 病人编号}
        MIMIC: 分三级
        1. 保存了官方数据集划分
            {'train':{patients}, 'val':{...}, 'test':{...}}
        2. 每个病人的检查以patient编码为key，保存在list中
            {'patient_id': [studys]}
        3. list中的检查按时间从早到晚保存
            {'study_id': ..., 'report': ..., 'path': ..., 'date':...}
        """
        with open(self.json_report, 'r') as f:
            self.report = json.load(f)

        self.pair_data = self.load_pair(subset)

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        raise NotImplementedError

    def load_pair(self, subset):
        raise NotImplementedError

    def sent2id(self, report_txt):
        report = []
        report.append(self.vocab('<start>'))
        report.extend([self.vocab(token) for token in report_txt.split()[:self.max_len - 2]])
        report.append(self.vocab('<end>'))
        report = torch.Tensor(report).long()
        return report


class IUSim1Dataset(_DatasetBase):
    """读取 1 similar 图像和报告
    json中存report信息
    pair_list存有配对的图像对文件名
    所有图像存在image_dir文件夹下面
    """

    def load_pair(self, subset):
        pair = pd.read_csv(self.pair_split)
        pair = pair[pair['split'] == subset]
        return [[prev, curr] for prev, curr in zip(pair['similar_0'], pair['filename'])]

    def __getitem__(self, index):
        prev_img_name, curr_img_name = self.pair_data[index]

        pre_image = Image.open(os.path.join(self.image_dir, prev_img_name)).convert('RGB')
        cur_image = Image.open(os.path.join(self.image_dir, curr_img_name)).convert('RGB')

        if self.transform is not None:
            pre_image = self.transform(pre_image)
            cur_image = self.transform(cur_image)

        images = torch.stack((pre_image, cur_image), 0)

        pre_uid, cur_uid = prev_img_name.split('_')[0], curr_img_name.split('_')[0]

        pre_report = self.sent2id(self.report[pre_uid])
        cur_report = self.sent2id(self.report[cur_uid])

        padd_pre_report = torch.zeros(self.max_len).long()
        padd_pre_report[:len(pre_report)] = pre_report

        return images, cur_report, index, curr_img_name, padd_pre_report


class IUSimNDataSet(_DatasetBase):
    '''读取N张 similar图像和报告
    json中存report信息
    pair_list存有配对的图像对文件名
    所有图像存在image_dir文件夹下面
    '''

    def load_pair(self, subset):
        pair = pd.read_csv(self.pair_split)
        pair = pair[pair['split'] == subset]

        pair_data = []
        # 先添加similar的图片，最后添加要生成报告的图片
        for i, row in pair.iterrows():
            items = []
            for j in range(self.N):
                items.append(row['similar_{}'.format(j)])
            items.append(row['filename'])
            pair_data.append(items)
        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data

    def __getitem__(self, index):
        names = self.pair_data[index]

        images, padd_pre_reports = [], []
        for i in range(self.N + 1):
            name = names[i]
            image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            # name: 377_IM-1889
            uid = name.split('_')[0]
            report = self.sent2id(self.report[uid])

            padd_pre_report = torch.zeros(self.max_len).long()
            padd_pre_report[:len(report)] = report

            images.append(image)
            padd_pre_reports.append(padd_pre_report)

        cur_report = report
        curr_name = name
        images = torch.stack(images, dim=0)
        # 为了代码简洁，循环中最后一步把current report添加进去了，这里去掉
        padd_pre_reports = torch.stack(padd_pre_reports[:-1], dim=0)

        return images, cur_report, index, curr_name, padd_pre_reports


class _MimicBase(_DatasetBase):
    """提供MIMIC数据集 getitem 功能"""
    def __getitem__(self, index):
        items = self.pair_data[index]

        images, padd_pre_reports = [], []
        for i in range(self.N + 1):
            # mimic 有 dicom 和 jpg 两种格式的图片，这里路径以dicom结尾，替换后缀
            name = os.path.splitext(items[i]['path'])[0] + '.jpg'
            image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            report = self.sent2id(items[i]['report'])
            padd_pre_report = torch.zeros(self.max_len).long()
            padd_pre_report[:len(report)] = report

            images.append(image)
            padd_pre_reports.append(padd_pre_report)

        cur_report = report
        curr_name = name
        images = torch.stack(images, dim=0)
        # 为了代码简洁，循环中最后一步把current report添加进去了，这里去掉
        padd_pre_reports = torch.stack(padd_pre_reports[:-1], dim=0)

        return images, cur_report, index, curr_name, padd_pre_reports

    @staticmethod
    def _load_id_report(report_data, subsets=None):
        if subsets is None:
            subsets = ['train', 'val', 'test']

        id_report = {}
        for subset in subsets:
            data = report_data[subset]
            for subject_id in data:
                studys = data[subject_id]
                for study in studys:
                    id_report[study['path']] = study['report']
        return id_report



class MimicNDropDataset(_MimicBase):
    """使用自身的历史报告作为condition, 最后1个作为target
    忽略condition数量少于N+1的study
    Note:
        N不同时，数据集大小不同
        对于val和test set, 存在标签泄露的问题，因为在预测时使用了其他test样本
    """

    def load_pair(self, subset):
        """
        report 分三级
        1. 保存了官方数据集划分
            {'train':{patients}, 'val':{...}, 'test':{...}}
        2. 每个病人的检查以patient编码为key，保存在list中
            {'patient_id': [studys]}
        3. list中的检查按时间从早到晚保存
            {'study_id': ..., 'report': ..., 'path': ..., 'date':...}
        """
        data = self.report

        pair_data = []
        for subject_id in data:
            studys = data[subject_id]
            # 需要前N次的检查，因此如果这个病人的检查数量少于N+1就不管这个病人
            # 最后1个作为target
            if len(studys) < self.N + 1:
                continue

            # 当检查数量大于N+1时，可形成多组
            # N=2, {0, 1, 2, 3} -> {0, 1, 2}, {1, 2, 3}
            for i in range(len(studys) - (self.N + 1)):
                items = []
                for j in range(self.N + 1):
                    item = {'path': studys[i + j]['path'], 'report': studys[i + j]['report']}
                    items.append(item)
                pair_data.append(items)

        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data


class MimicNSupSimDataset(_MimicBase):
    """使用自身的历史报告作为condition, 最后1个作为target
    condition数量少于N+1的 study 选取similar图片作为补充
    Note:
        N不同时，数据集大小相同
        对于val和test set, 存在标签泄露的问题，因为在预测时使用了其他test样本
    """

    # 病人的检查数量少于N+1时，读取similar文件
    def load_pair(self, subset):
        report_data = self.report
        data = report_data[subset]

        # pair_filename中保存根据图片similar计算得到的simlar pair
        pair = pd.read_csv(self.pair_split).set_index('filename')
        # similar 来自training set，load 对应的 report
        id_report = self._load_id_report(report_data, ['train'])

        pair_data = []
        for subject_id in data:
            studys = data[subject_id]

            # 先构造好一个病人的一串condition
            check_list = []
            # 以第一次 study 前的 N 次 similar 作为补充condition
            first_study_id = studys[0]['path']

            # 添加用第一次 study 的 similar
            for i in range(self.N):
                similar_id = pair.at[first_study_id, f'similar_{self.N - 1 - i}']
                report = id_report[similar_id]
                check_list.append({'path': similar_id, 'report': report})

            # 添加病人自己的历史检查记录
            for study in studys:
                check_list.append({'path': study['path'], 'report': study['report']})

            # 最后1个作为target，形成多组
            # {0, 1, 2, 3} -> {0, 1, 2}, {1, 2, 3}
            for i in range(len(check_list) - (self.N)):
                pair_data.append(check_list[i: i + (self.N + 1)])

        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data


class MimicNSupRandomDataset(_MimicBase):
    """使用自身的历史报告作为condition, 最后1个作为target
    condition数量少于N+1的 study 从training set中随机选取一例作为补充
    Note:
        N不同时，数据集大小相同
        对于val和test set, 存在标签泄露的问题，因为在预测时使用了其他test样本
    """

    def load_pair(self, subset):
        import random
        pair_data = self.report
        data = pair_data[subset]
        train_data = pair_data['train']

        pair_data = []
        for subject_id in data:
            studys = data[subject_id]
            # 需要前N次的检查，因此如果这个病人缺少第N次检查前的报告就从训练集随机取一个
            check_list = []
            for train_subject_id in random.sample(list(train_data), self.N):
                select_list = train_data[train_subject_id]
                check_list.append({'path': select_list[0]['path'], 'report': select_list[0]['report']})

            # 添加病人自己的历史检查记录
            for study in studys:
                check_list.append({'path': study['path'], 'report': study['report']})

            # 最后1个作为target，可形成多组
            # {0, 1, 2, 3} -> {0, 1, 2}, {1, 2, 3}
            for i in range(len(check_list) - (self.N)):
                pair_data.append(check_list[i: i + (self.N + 1)])

        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data


class MimicNSimDataset(_MimicBase):
    """全部选取similar作为condition, 自身作为target
    Note:
        N不同时，数据集大小相同
        不存在标签泄露的问题，similar均来自training set
    """

    # 读取similar文件
    def load_pair(self, subset):
        self.report = self._load_id_report(self.report)

        pair = pd.read_csv(self.pair_split)
        pair = pair[pair['split'] == subset]

        pair_data = []
        for i, row in pair.iterrows():
            items = []
            for j in range(self.N):
                similar_id = row[f'similar_{j}']
                items.append({'path': similar_id, 'report': self.report[similar_id]})
            items.append({'path': row['filename'], 'report': self.report[row['filename']]})
            pair_data.append(items)

        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data


class MimicNSimDatasetRetrieval5(_MimicBase):
    """全部选取similar作为condition, 自身作为target
    Note:
        N不同时，数据集大小相同
        不存在标签泄露的问题，similar均来自training set
    """

    # 读取similar文件
    def load_pair(self, subset):
        self.report = self._load_id_report(self.report)

        pair = pd.read_csv(self.pair_split)
        pair = pair[pair['split'] == subset]

        pair_data = []
        for i, row in pair.iterrows():
            items = []
            for j in range(10):
                similar_id = row[f'similar_{j}']
                items.append({'path': similar_id, 'report': self.report[similar_id]})
            items.append({'path': row['filename'], 'report': self.report[row['filename']]})
            pair_data.append(items)

        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data


    def __getitem__(self, index):
        items = self.pair_data[index]

        cur_name = os.path.splitext(items[-1]['path'])[0] + '.jpg'
        cur_image = Image.open(os.path.join(self.image_dir, cur_name)).convert('RGB')
        if self.transform is not None:
            cur_image = self.transform(cur_image)
        cur_report = self.sent2id(items[-1]['report'])

        pre_name = os.path.splitext(items[4]['path'])[0] + '.jpg'
        pre_image = Image.open(os.path.join(self.image_dir, pre_name)).convert('RGB')
        if self.transform is not None:
            pre_image = self.transform(pre_image)
        pre_report = self.sent2id(items[4]['report'])
        padd_pre_report = torch.zeros(self.max_len).long()
        padd_pre_report[:len(pre_report)] = pre_report

        images = torch.stack([pre_image, cur_image], dim=0)
        # 为了代码简洁，循环中最后一步把current report添加进去了，这里去掉
        padd_pre_reports = torch.stack([padd_pre_report], dim=0)

        return images, cur_report, index, cur_name, padd_pre_reports


class MimicNSimDatasetRetrieval10(MimicNSimDatasetRetrieval5):
    """全部选取similar作为condition, 自身作为target
    Note:
        N不同时，数据集大小相同
        不存在标签泄露的问题，similar均来自training set
    """

    def __getitem__(self, index):
        items = self.pair_data[index]

        cur_name = os.path.splitext(items[-1]['path'])[0] + '.jpg'
        cur_image = Image.open(os.path.join(self.image_dir, cur_name)).convert('RGB')
        if self.transform is not None:
            cur_image = self.transform(cur_image)
        cur_report = self.sent2id(items[-1]['report'])

        pre_name = os.path.splitext(items[9]['path'])[0] + '.jpg'
        pre_image = Image.open(os.path.join(self.image_dir, pre_name)).convert('RGB')
        if self.transform is not None:
            pre_image = self.transform(pre_image)
        pre_report = self.sent2id(items[9]['report'])
        padd_pre_report = torch.zeros(self.max_len).long()
        padd_pre_report[:len(pre_report)] = pre_report

        images = torch.stack([pre_image, cur_image], dim=0)
        # 为了代码简洁，循环中最后一步把current report添加进去了，这里去掉
        padd_pre_reports = torch.stack([padd_pre_report], dim=0)

        return images, cur_report, index, cur_name, padd_pre_reports


class Mimic4LastDataset(_MimicBase):
    """选取有4次及以上study的病人作为数据集，不管N为几，均使用最后一次作为target
    Note:
        N <= 3
        不存在标签泄露问题
        此数据集是原数据集的子集，不能直接和其他方法比较
        与其他方法比较时，为保证公平，需要将val/test set的 condition
          作为其他方法的 training 数据
    """
    def load_pair(self, subset):
        report_data = self.report
        data = report_data[subset]

        pair_data = []
        for subject_id in data:
            studys = data[subject_id]
            check_list = []

            # 添加病人自己的历史检查记录
            for study in studys:
                check_list.append({'path': study['path'], 'report': study['report']})

            # 跳过少于4次的病人
            if len(check_list) < 3 + 1:
                continue
            # 只取最后1个作为target，如N=2时
            # {0, 1, 2, 3} -> {1, 2, 3}
            pair_data.append(check_list[-(self.N + 1):])

        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data


class CovidDataSet(_DatasetBase):
    """Covid数据集, 采用前一次检查作为condition
       中文采取按字分词的方法
       covid 95% case的finding部分长度最大为108，因此设置最大长度为110
    """
    def load_pair(self, subset):
        pair = pd.read_csv(self.pair_split)
        pair = pair[pair['split'] == subset]
        pair_data = [[prev, curr] for prev, curr in zip(pair['similar_0'], pair['filename'])]
        print('=======>> Load {} dataset {} items <<=========='.format(subset, len(pair_data)))
        return pair_data

    # 中文按字分词，不用split
    def sent2id(self, report_txt):
        report = []
        report.append(self.vocab('<start>'))
        report.extend([self.vocab(token) for token in report_txt[:self.max_len - 2]])
        report.append(self.vocab('<end>'))
        report = torch.Tensor(report).long()
        return report

    def __getitem__(self, index):
        prev_img_name, curr_img_name = self.pair_data[index]

        pre_image = Image.open(os.path.join(self.image_dir, prev_img_name + '.png')).convert('RGB')
        cur_image = Image.open(os.path.join(self.image_dir, curr_img_name + '.png')).convert('RGB')

        if self.transform is not None:
            pre_image = self.transform(pre_image)
            cur_image = self.transform(cur_image)

        images = torch.stack((pre_image, cur_image), 0)

        pre_report = self.sent2id(self.report[prev_img_name])
        cur_report = self.sent2id(self.report[curr_img_name])

        padd_pre_report = torch.zeros((1, self.max_len)).long()
        padd_pre_report[:, :len(pre_report)] = pre_report

        return images, cur_report, index, curr_img_name, padd_pre_report


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, report).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, report).
            - image: torch tensor of shape (3, 256, 256).
            - report: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        img_ids: image ids in COCO dataset, for evaluation purpose
        filenames: image filenames in COCO dataset, for evaluation purpose
    """

    # Sort a data list by report length (descending order).
    # sort方法原地排序， sorted构建新的输出
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, img_ids, filenames, prev_repos = zip(*data)  # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    img_ids = list(img_ids)
    filenames = list(filenames)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros((len(captions), max(lengths))).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap

    # condition已经在之前padding为0了，直接stack
    prev_repos = torch.stack(prev_repos, 0)

    return images, targets, lengths, img_ids, filenames, prev_repos


def get_loader(image_dir, json_report, csv_pair_split, vocab, transform=None, batch_size=60, shuffle=True, num_workers=4,
               type='single', N=1, max_len=80, subset='train'):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if type == 'iu':
        setname = IUSimNDataSet
    elif type == 'covid':
        setname = CovidDataSet
    elif type == 'mimicsim':
        setname = MimicNSimDataset
    elif type == 'mimic4last':
        setname = Mimic4LastDataset
    elif type == 'mimicR5':
        setname = MimicNSimDatasetRetrieval5
    elif type == 'mimicR10':
        setname = MimicNSimDatasetRetrieval10
    else:
        raise NameError

    dataset = setname(image_dir=image_dir, json_report=json_report, pair_split=csv_pair_split, vocab=vocab, N=N,
                      max_len=max_len, subset=subset, transform=transform)

    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              pin_memory=True)
    return data_loader


if __name__ == '__main__':
    import pickle
    from torchvision import transforms
    from config import opts
    args = opts.parse_opt()

    subsets = ['test', 'train', 'val']
    pair_filename = args.pair_list
    pair_data = pd.read_csv(pair_filename)

    root = args.image_dir
    json_path = args.caption_json
    vocab_path = args.vocab_path

    # Load vocabulary wrapper.
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    for subset in subsets:
        print(subset)
        data_loader = get_loader(root, json_path, pair_filename, vocab, transform=transform, shuffle=False,
                                 type=args.dataset, subset=subset)
        # for i, (images, target, lengths, _, image_ids, prev_repo) in tqdm(enumerate(data_loader)):
        for i, (images, target, lengths, _, image_ids, prev_repo) in enumerate(data_loader):
            for image_id in image_ids:
                split = pair_data[pair_data['filename'] == image_id]['split'].item()
                simi_id = pair_data[pair_data['filename'] == image_id]['similar_0'].item()
                simi_set = pair_data[pair_data['filename'] == simi_id]['split'].item()
                # if pair_data[pair_data['filename'] == image_id]['split'].item() != subset:
                print(' | '.join([image_id, simi_id, subset, split, simi_set]))

import argparse
import os
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = len(CLASS_NAMES)
CKPT_PATH = 'model.pth.tar'


class DenseNet121(nn.Module):
    """Model modified.
    Adapted from https://github.com/arnoweng/CheXNet
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121()
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def rename_layer(checkpoint):
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    #     checkpoint = torch.load('./model.pth.tar')
    state_dict = checkpoint['state_dict']
    remove_data_parallel = True  # Change if you don't want to use nn.DataParallel(model)

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]
        # Delete old key only if modified.
        if match or remove_data_parallel:
            del state_dict[key]
    checkpoint['state_dict'] = state_dict
    return checkpoint


def main(args):
    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        checkpoint = rename_layer(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint", CKPT_PATH)
    else:
        print("=> no checkpoint found")

    model = model.cuda()
    model.eval()

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    tfms = transforms.Compose([transforms.Resize(256),
                               transforms.TenCrop(224),
                               transforms.Lambda
                               (lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
                               transforms.Lambda
                               (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                               ])

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    patient_files = [pf for pf in os.listdir(args.data) if not pf.startswith('.')]
    already_save = [pf for pf in os.listdir(args.output)]
    with torch.no_grad():
        for patient_file in tqdm(patient_files):
            patient_name = patient_file[:-4]

            if patient_name + '.npy' in already_save:
                continue

            images = tfms(Image.open(os.path.join(args.data, patient_file)).convert('RGB'))

            n_crops, c, h, w = images.size()
            input_var = images.view(-1, c, h, w).cuda()

            # 10 * N_CLASSES
            tag_features = model(input_var)
            tag_features_mean = tag_features.mean(0)

            # 10 * 1024 * 7 * 7(if input 224 * 224)(if 512 * 512 -> 16 * 16)
            features = model.densenet121.features(input_var)
            features_mean = features.mean(0)

            # 7 * 7 * 1024
            feature = features_mean.cpu().detach().numpy().transpose((1, 2, 0))
            # 14,
            tag_feature = tag_features_mean.cpu().detach().numpy()

            # 49 * 1024
            n_channel = feature.shape[-1]
            feature = feature.reshape((-1, n_channel))
            np.save(os.path.join(args.output, patient_name + '.npy'), feature)
            np.save(os.path.join(args.output, patient_name + '_tag.npy'), tag_feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default="data/",
                        help="the begin directory of all")
    parser.add_argument('-d', '--data', type=str, default="images/images_normalized",
                        help="image file path")
    parser.add_argument('-o', '--output', type=str, default="feature",
                        help="output feature directory")
    parser.add_argument('-n', '--num_class', type=int, default=14,
                        help="pretrain model classes")
    parser.add_argument('-g', '--gpu', type=str, default="7",
                        help="which gpu to use")

    args = parser.parse_args()
    args.data = os.path.join(args.root, args.data)
    args.output = os.path.join(args.root, args.output)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)

    main(args)

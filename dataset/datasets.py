# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from builtins import getattr
import os
import torch
import math

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform


from dataset.dataset_lmdb import ImageLmdb
from dataset.concatdatasets import ConcatDataset


def build_dataset(is_train, args):


    transform = RegularTransform(is_train, args)
    num_view = getattr(args, 'num_view', 1)
    

    font_path = getattr(args, 'font_path', None)


    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    elif isinstance(transform, RegularTransform):
        for t in transform.transforms.transforms:
            print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")


    if args.data_set == 'image_lmdb':
        root = args.data_path if is_train else args.eval_data_path
        print(root)
        if isinstance(root, list):
            dataset_list = []
            for data_path in root:
                dataset = ImageLmdb(data_path, args.voc_type, args.max_len,
                    args.num_samples if is_train else math.inf, transform=transform,
                    use_aug=(num_view>1. and is_train),
                    font_path=font_path, use_class_binary_sup= is_train, args=args)
                dataset_list.append(dataset)
            dataset = ConcatDataset(dataset_list)
        else:
            dataset = ImageLmdb(root, args.voc_type, args.max_len,
                        args.num_samples if is_train else math.inf, transform=transform,
                        use_aug=(num_view>1. and is_train),
                        font_path=font_path, use_class_binary_sup=is_train, args=args)
        nb_classes = len(dataset.classes)
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    mean = std = 0.5
    t = []
    t.append(transforms.Resize((args.input_h, args.input_w), interpolation=3))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class RegularTransform(object):
    def __init__(self, is_train, args):
        mean = std = 0.5

        self.transforms = transforms.Compose([
            transforms.Resize((args.input_h, args.input_w), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        self.input_h, self.input_w = args.input_h, args.input_w

    def __call__(self, image):
        return self.transforms(image)
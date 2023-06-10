

import os

ROOT_DATASET = '/home/lthpc/yanglinfeng/dataset/'


def return_ucf101(modality):
    filename_categories = 101
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        #root_data = ROOT_DATASET + 'UCF101/'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = '{:d}.jpg'
        # prefix = '{:05d}.jpg'  #win专用
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'hmdb51/jpg'
        filename_imglist_train = 'hmdb51/file_list/train_split_3.txt'
        filename_imglist_val = 'hmdb51/file_list/val_split_3.txt'
        prefix = '{:d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_diving48(modality):
    filename_categories = 48
    if modality == 'RGB':
        prefix = '{:d}.jpg'
        root_data = ROOT_DATASET + 'diving48/jpgs'
        filename_imglist_train = 'diving48/file_list/trainlist01.txt'
        filename_imglist_val = 'diving48/file_list/testlist01.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_tinykinetics(modality):
    filename_categories = 200
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'tiny-kinetics'
        filename_imglist_train = 'tiny-kinetics/file_list/train_split.txt'
        filename_imglist_val = 'tiny-kinetics/file_list/test_split.txt'
        prefix = '{:d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'diving48': return_diving48, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'tinykinetics': return_tinykinetics }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix

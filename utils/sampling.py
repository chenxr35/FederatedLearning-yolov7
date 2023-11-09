import numpy as np
import collections


def voc_iid(dataset, num_users):
    """
    Sample I.I.D. client data from PASCAL VOC dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(dataset.__len__()/num_users)
    dict_users, all_idxs = {}, [i for i in range(dataset.__len__())]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def voc_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from PASCAL VOC dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dataset.generate_class_image()
    dict_users = collections.defaultdict(set)
    allocated = set()
    i = 0
    for class_id in dataset.class_image:
        class_idxs = dataset.get_class_image(class_id)
        class_idxs = set(class_idxs) - allocated
        allocated = allocated.union(class_idxs)
        class_idxs = list(class_idxs)
        id_1 = i % num_users
        id_2 = (i+1) % num_users
        selection = set(np.random.choice(class_idxs, int(len(class_idxs)//2), replace=False))
        dict_users[id_1] = dict_users[id_1].union(selection)
        remainder = set(class_idxs) - selection
        dict_users[id_2] = dict_users[id_2].union(remainder)
        i += 1
    return dict_users
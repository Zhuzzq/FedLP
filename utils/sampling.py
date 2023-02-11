import numpy as np
from torchvision import datasets, transforms
from collections import Counter
import matplotlib.pyplot as plt

"""
Returns train and test datasets and a user group which is a dict where
the keys are the user index and the values are the corresponding data for
each of those users.

dict users {0: array([], dtype=int64), 1: array([], dtype=int64), 2....}
"""

def get_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def get_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 200, 300
    num_shards, num_imgs = 2*num_users, int(len(dataset) / (2*num_users))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    labels = np.array(dataset.targets)

    # sort labels
    idxs = labels.argsort()

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def get_mixed_noniid(dataset, num_users, ratio):
    """
    Sample mixed non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :non-i.i.d ratio
    :return:
    """
    # iid sample
    iid_num = int(len(dataset) * (1 - ratio))
    print(iid_num)

    iidnum_items = int(iid_num / num_users)
    dict_users, all_idxs = {}, [i for i in range(iid_num)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, iidnum_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = np.array(list(dict_users[i]))

    # non-iid sample
    # num_items = int(num_shards / num_users)
    num_items = 2
    num_imgs = len(dataset)//num_users//num_items - iidnum_items // num_items  # mnist 300? cifar 500
    # num_shards = int(noniid_num / num_imgs)
    num_shards = int(len(dataset) / num_imgs)
    idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets[0:num_shards * num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_items, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def get_dirichlet_noniid(dataset, num_users, alpha=1.0, sample_num=0):
    if sample_num == 0:
        num_items = len(dataset)
    else:
        num_items = sample_num
    labels = dataset.targets[0:num_items]
    num_cls = max(labels) + 1
    idx = [np.argwhere(np.array(labels) == y).flatten() for y in range(num_cls)]
    # print(np.argwhere(labels == 1))
    label_distribution = np.random.dirichlet([alpha] * num_users, num_cls)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for c, fracs in zip(idx, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            dict_users[i] = np.concatenate((dict_users[i], idcs), axis=0)
            # client_idcs[i] += [idcs]

    # client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return dict_users


def dataset_stats(dict_users, dataset, args):
    # dict users {0: array([], dtype=int64), 1: array([], dtype=int64), ..., 100: array([], dtype=int64)}
    stats = {i: np.array([], dtype='int64') for i in range(len(dict_users))}
    for key, value in dict_users.items():
        for x in value:
            stats[key] = np.concatenate((stats[key], np.array([dataset[x][1]])), axis=0)
    
    nparray = np.zeros([args.num_classes, args.num_users], dtype = int)
    for j in range(args.num_users):
        cls = stats[j]
        cls_counter = Counter(cls)
        for i in range(args.num_classes):
            nparray[i][j] = cls_counter[i]

    fig, ax = plt.subplots()
    bottom = np.zeros([args.num_users], dtype=int)
    for cls in range(args.num_classes):
        ax.bar(range(args.num_users), nparray[cls], bottom=bottom, label='class{}'.format(cls))
        bottom += nparray[cls]
    ax.legend(loc='lower right')
    plt.title('Data Distribution')
    plt.xlabel('Clients')
    plt.ylabel('Amount of Training Data')
    plt.show()


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = get_dirichlet_noniid(dataset_train, num, alpha=0.1, sample_num=1000*num)

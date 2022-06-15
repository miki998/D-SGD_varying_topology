import numpy as np


class Loader:
    """
    Loader of MNIST dataset separating labels and images
    """
    
    def __init__(self):
        pass

    def toArray(self, dataset):
        array  = []
        labels = []
        for img,l in dataset:
            array.append(np.array(img))
            labels.append(l)

        return np.asarray(array), np.asarray(labels)

# source: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/sampling.py
def mnist_noniid(dataset, label, num_users, num_shards=20, num_imgs=2000):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = label.clone().numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
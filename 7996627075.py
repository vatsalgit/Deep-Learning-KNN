import sys
import os
import pickle

class Dlknn (object):

    def __init__(self, K, N, D, PATH_TO_DATA, **kwargs):
        self.K = K
        self.N = N
        self.D = D
        self.PATH_TO_DATA = PATH_TO_DATA


    def load_images(self, num=1000):
        with open(self.PATH_TO_DATA, 'rb') as fo:
            my_dict = pickle.load(fo, encoding='bytes')
        labels = my_dict[b'labels'][:num]
        image_data = my_dict[b'data'][:num]
        return labels, image_data


def load_args(args='sys'):
    if args == 'sys':
        return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    else:
        return 3, 2, 100, "C:/Users/vatsal/OneDrive/Deep Learning/Data/data_batch_1"


if __name__ == '__main__':
    k, d, n, PATH_TO_DATA = load_args('default')
    knn = Dlknn(k, d, n, PATH_TO_DATA)
    labels, image_data = knn.load_images()
    main()

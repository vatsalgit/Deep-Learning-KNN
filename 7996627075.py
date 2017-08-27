import sys
import numpy as np
import pickle
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

class Dlknn (object):

    def __init__(self, K, D, N, PATH_TO_DATA, **kwargs):
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

    def train_test_split(self):
        labels, image_data = self.load_images()
        train_X, train_y = image_data[self.N:], labels[self.N:]
        test_X, test_y = image_data[:self.N], labels[:self.N]
        return train_X, train_y, test_X, test_y

    @staticmethod
    def convert_to_grayscale(X):
        new_image = []
        for image in X:
            R , G, B = image[:1024], image[1024:2048], image[2048:]
            gray = (0.299 * R) + (0.587 * G) + (0.114 * B)
            new_image.append(gray)
        return new_image

    def grayscale_transform(self):
        train_X, train_y, test_X, test_y = self.train_test_split()
        train_X = self.convert_to_grayscale(train_X)
        test_X = self.convert_to_grayscale(test_X)
        return np.array(train_X), np.array(train_y), test_X, test_y

    def pca_fit(self, X):
        pca = PCA(n_components=self.D, svd_solver='full')
        self.fitted_pca = pca.fit(X)

    def pca_transform(self, X):
        new_X = self.fitted_pca.transform(X)
        return new_X

    def pca_fit_transform(self, train_X, test_X):
        self.pca_fit(train_X)
        pca_train = self.pca_transform(train_X)
        pca_test = self.pca_transform(test_X)
        return pca_train, pca_test

    def calculate_inverse_euclidean(self, train_X, test_image, train_y):
        distances = [(1/euclidean(test_image, n[0]) , n[1]) for n in zip(train_X, train_y) ]
        distances = sorted(distances, key=lambda x:x[0], reverse=True)[:self.K]
        return distances

    def predict(self, train_X, test_X, train_y, test_y):
        all_image_distances = [(self.calculate_inverse_euclidean(train_X, test_image, train_y), label) for test_image,label in zip(test_X, test_y) ]
        myfile = open('7996627075.txt', 'w')
        for image, true_label in all_image_distances:
            my_dict = {}
            for distance,label in image:
                my_dict[label] = my_dict.get(label, 0 ) + distance
            predicted_label = max(my_dict, key=my_dict.get)
            output_string = str(predicted_label) + ' ' + str(true_label) +'\n'
            myfile.write(output_string)
        myfile.close()

def load_args(args='sys'):
    if args == 'sys':
        return int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    else:
        return 5, 100, 10, "C:/Users/vatsal/OneDrive/Deep_Learning/Data/data_batch_1"

def main():
        k, d, n, PATH_TO_DATA = load_args('sys')
        knn = Dlknn(K=k, D=d, N=n, PATH_TO_DATA=PATH_TO_DATA)
        train_X, train_y, test_X, test_y = knn.grayscale_transform()
        train_X, test_X = knn.pca_fit_transform(train_X, test_X)
        knn.predict(train_X, test_X, train_y, test_y)

if __name__ == '__main__':
    main()

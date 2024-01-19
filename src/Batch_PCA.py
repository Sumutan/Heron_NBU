import numpy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
import time

from mindspore import Tensor
import mindspore.ops as ops
from mindspore.scipy.linalg import eigh


class Batch_PCA:
    def __init__(self, n_components=32):
        self.n_components = n_components

    def batch_fit_transform(self, X: numpy.ndarray):
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]
        assert len(X.shape) == 3, f'dimension error, expect 3, get {X.shape}'

        Y = []
        for X_i in X:
            data = X_i.T
            pca = PCA(n_components=self.n_components)
            output = pca.fit_transform(data).T
            Y.append(output)
        Y = np.array(Y).squeeze()
        return Y

    def fit_transform(self, X: numpy.ndarray):  # [1568,768]->[32,768]
        assert len(X.shape) == 2

        data = X.T  # [768,1568]
        pca = PCA(n_components=self.n_components)
        output = pca.fit_transform(data).T

        return output  # [32,768]


class Batch_PCA_np(Batch_PCA):
    def __init__(self, n_components=32):
        super().__init__(n_components)

    def fit_transform(self, X: numpy.ndarray):
        X = X.T  # 对样本进行融合，而不是对特征降纬，所以先交换B，C对纬度 ->C，B
        mean = np.mean(X, axis=0)  # 计算均值
        X_centered = X - mean  # 将数据减去均值
        cov_matrix = np.cov(X_centered, rowvar=False)  # 计算协方差矩阵
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # 特征值分解
        # 选择主成分
        sorted_indices = np.argsort(eigenvalues)[::-1]  # 按特征值降序排序的索引
        selected_indices = sorted_indices[:self.n_components]  # 选择前n个特征值最大的索引
        selected_eigenvectors = eigenvectors[:, selected_indices]

        # 将数据投影到主成分上
        X_pca = np.dot(X_centered, selected_eigenvectors)
        return X_pca.T


class Batch_PCA_ms(Batch_PCA):
    def __init__(self, n_components=32):
        super().__init__(n_components)

    def fit_transform(self, X: numpy.ndarray):
        X_tensor = Tensor(X.astype(np.float32)).T  # B,C->C,B

        mean = ops.mean(X_tensor, axis=0)  # 计算均值
        X_centered = X_tensor - mean  # 将数据减去均值以获得零均值数据
        cov_matrix = ops.cov(X_centered.T)  # 计算协方差矩阵  # ms2.x support ops.cov #ops.cov不含rowvar，所以手动转置输入矩阵

        eigvals, eigvecs = eigh(cov_matrix)  # 特征值分解

        # 选择主成分
        sorted_indices = ops.argsort(eigvals)[::-1]  # 按特征值降序排序的索引
        selected_indices = sorted_indices[:self.n_components]  # 选择前2个特征值最大的索引
        selected_eigvecs = eigvecs[:, selected_indices]
        # 将数据投影到主成分上
        X_pca = ops.dot(X_centered, selected_eigvecs)

        return X_pca.astype(np.float32)


class Batch_KMeans:
    def __init__(self, n_components=16, max_iter=300, batch_size=None):
        self.n_components = n_components
        self.max_iter = max_iter
        if batch_size is None:
            self.KMeansMethod = 'KMeans'
            self.kmeans = KMeans(n_clusters=self.n_components, max_iter=self.max_iter)
        else:
            self.KMeansMethod = 'MiniBatchKMeans'
            self.batch_sise = batch_size
            self.kmeans = MiniBatchKMeans(n_clusters=self.n_components, max_iter=self.max_iter,
                                          batch_size=self.batch_sise)

    def batch_fit_transform(self, X: numpy.ndarray):
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]
        assert len(X.shape) == 3, f'dimension error, expect 3, get {X.shape}'

        Y = []
        for X_i in X:
            data = X_i
            self.kmeans.fit(data)
            centers = self.kmeans.cluster_centers_
            # output = np.mean(centers, axis=0, keepdims=True)
            Y.append(centers)
        Y = np.array(Y)
        return Y

    def fit_transform(self, X: numpy.ndarray):  # [1568,768]->[32,768]
        assert len(X.shape) == 2
        data = X  # [1568,768]

        self.kmeans.fit(data)
        centers = self.kmeans.cluster_centers_
        output = np.mean(centers, axis=0, keepdims=False)
        print(output.shape)

        return output  # [768]


def demo():
    N, C = 1568, 768
    data = np.random.rand(N, C)

    start = time.time()

    data = data.T
    # 计算PCA
    pca = PCA(n_components=32)
    output = pca.fit_transform(data).T

    end1 = time.time()

    # 转换数据
    transformed_data = pca.transform(data)
    transformed_data = transformed_data.T

    end2 = time.time()

    print("Transformed Data:")
    print(transformed_data.shape)

    print("PCA fitting Time:", end1 - start)
    print("PCA transform Time:", end2 - end1)


def PCA_test_sklearn(X: numpy.ndarray):  # no batch

    for i in range(5):
        start = time.time()

        pca = Batch_PCA(n_components=32)
        # output=pca.batch_fit_transform(data)
        output = pca.batch_fit_transform(data)

        end = time.time()
        print(f"step {i} : Transformed Data:{output.shape}  PCA fit_transform Time:{end - start}")


def PCA_test_numpy(X: numpy.ndarray):  # no batch
    for i in range(5):
        start = time.time()

        pca = Batch_PCA_np(n_components=32)
        output = pca.fit_transform(data)

        end = time.time()
        print(f"step {i} : np_Transformed Data:{output.shape}  PCA fit_transform Time:{end - start}")


def PCA_test_ms(X: numpy.ndarray):  # no batch
    for i in range(5):
        start = time.time()

        pca = Batch_PCA_ms(n_components=32)
        output = pca.fit_transform(data)

        end = time.time()
        print(f"step {i} : np_Transformed Data:{output.shape}  PCA fit_transform Time:{end - start}")


def Kmeans_test(X: numpy.ndarray):  # no batch
    for i in range(5):
        start = time.time()

        kmeans = Batch_KMeans(n_components=4, max_iter=300,batch_size=64)
        output = kmeans.batch_fit_transform(data)

        end = time.time()
        print(f"step {i} : np_Transformed Data:{output.shape}  PCA fit_transform Time:{end - start}")


if __name__ == '__main__':
    B, N, C = 4, 1568, 768
    data = np.random.rand(B, N, C)
    # N, C = 1568, 768
    # data = np.random.rand(N, C)

    print("strat PCA_test_sklearn:")
    PCA_test_sklearn(data)
    #
    # print("strat PCA_test_numpy:")
    # PCA_test_numpy(data)
    #
    # print("strat PCA_test_ms:")
    # PCA_test_ms(data)

    print("strat KMeans_test:")
    Kmeans_test(data)

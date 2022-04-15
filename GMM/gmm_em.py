# coding: utf-8
# 高斯混合模型 使用EM算法求解

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    dataset = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split(' ')
        fltline = list(map(float, curline))
        dataset.append(fltline)
    return dataset


def prob(x, mu, sigma):
    n = np.shape(x)[1]
    expOn = float(-0.5 * (x - mu) * (sigma.I) * (x - mu).T)
    divBy = pow(2 * np.pi, n / 2) * pow(np.linalg.det(sigma), 0.5)
    return pow(np.e, expOn) / divBy


# EM
def EM(dataMat, maxIter=50):
    m, n = np.shape(dataMat)
    alpha = [1/3, 1/3, 1/3]
    mu = [dataMat[5, :], dataMat[21, :], dataMat[26, :]]
    sigma = [np.mat([[0.1, 0], [0, 0.1]]) for i in range(3)]
    gamma = np.mat(np.zeros((m, 3)))
    for i in range(maxIter):
        for j in range(m):
            sumAlphaMulP = 0
            for k in range(3):
                gamma[j, k] = alpha[k] * prob(dataMat[j, :], mu[k], sigma[k])
                sumAlphaMulP += gamma[j, k]
            for k in range(3):
                gamma[j, k] /= sumAlphaMulP
        sumGamma = np.sum(gamma, axis=0)

        for k in range(3):
            mu[k] = np.mat(np.zeros((1, n)))
            sigma[k] = np.mat(np.zeros((n, n)))
            for j in range(m):
                mu[k] += gamma[j, k] * dataMat[j, :]
            mu[k] /= sumGamma[0, k]
            for j in range(m):
                sigma[k] += gamma[j, k] * (dataMat[j, :] - mu[k]).T * (dataMat[j, :] - mu[k])
            sigma[k] /= sumGamma[0, k]
            alpha[k] = sumGamma[0, k] / m
    return gamma


def initCentroids(dataMat, k):
    numSamples, dim = dataMat.shape
    centroids = np.zeros((k ,dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        centroids[i, :] = dataMat[index, :]
    return centroids


def gaussianCluster(dataMat):
    m, n = np.shape(dataMat)
    centroids = initCentroids(dataMat, m)
    clusterAss = np.mat(np.zeros((m, 2)))
    gamma = EM(dataMat)
    for i in range(m):
        clusterAss[i, :] = np.argmax(gamma[i, :]), np.amax(gamma[i, :])
    for j in range(m):
        pointsInCluster = dataMat[np.nonzero(clusterAss[:, 0].A == j)[0]]
        centroids[j, :] = np.mean(pointsInCluster, axis=0)
    return centroids, clusterAss


def showCluster(dataMat, k, centroids, clusterAss):
    numSamples, dim = dataMat.shape
    if dim != 2:
        return
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAss[i, 0])
        plt.plot(dataMat[i, 0], dataMat[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


if __name__ == "__main__":
    dataMat = np.mat(load_data('watermelon.txt'))
    centroids, clusterAssign = gaussianCluster(dataMat)
    print(clusterAssign)
    showCluster(dataMat, 3, centroids, clusterAssign)
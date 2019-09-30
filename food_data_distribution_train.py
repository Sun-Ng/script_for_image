import os
import cv2
import math
import tqdm
import random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ranksums
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.rc("font",family="SimHei",size="15")  #解决中文乱码问题

def CosineDistance(matrix1, matrix2):
    '''Calculate Cosine Distance between two matrix'''

    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(
        matrix1_matrix2,
        np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance


def EuclideanDistances(A, B):
    '''Calculate Euclidean Distance between two matrix'''

    BT = B.transpose()
    vecProd = np.dot(A, BT)

    SqA = A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))

    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


# matrix1 = np.array([[1, 1], [1, 2]])
# matrix2 = np.array([[2, 1], [2, 2], [2, 3]])

# Cosine_dis1 = CosineDistance(matrix1, matrix2)
# Cosine_dis2 = cosine_similarity(matrix1, matrix2)
# print('Cosine_dis1:', Cosine_dis1)
# print('Cosine_dis2:', Cosine_dis2)

# Euclidean_dis1 = EuclideanDistances(matrix1, matrix2)
# Euclidean_dis2 = cdist(matrix1, matrix2, metric='euclidean')
# print ('Euclidean_dis1:', Euclidean_dis1)
# print ('Euclidean_dis2:', Euclidean_dis2)

feature_path = "/data1/pub/zhengjie_pub/food_feature/"
# feature_train = np.load(feature_path + "feature_train.npy", allow_pickle=True)
# feature_val = np.load(feature_path + "feature_val.npy", allow_pickle=True)
feature_test = np.load(feature_path + "feature_test.npy", allow_pickle=True)

# class_number = len(feature_test)
# for c in range(class_number):
#     img_num = feature_test[c].shape[0]
#     print(img_num)

sample = 10
class_number = len(feature_test)

intra_euclidean_dis_list = []
for c in tqdm.tqdm(range(class_number)):
    num_img = feature_test[c].shape[0]
    print('cur_class_num:', num_img) 

    feature = np.squeeze(feature_test[c])
#     print('feature shape:', feature.shape)

    if num_img <= sample:
        num_sample = num_img
        sample_feature = feature
    else:
        num_sample = sample
        random_list = random.sample(list(range(num_img)), num_sample)
        sample_feature = feature[random_list]

    assert sample_feature.shape[0] <= num_sample, 'Error Sample Number'

    Euclidean_dis = cdist(sample_feature, sample_feature, metric='euclidean')
    cur_euclidean_list = list(Euclidean_dis[np.triu_indices(num_sample, 1)])
    intra_euclidean_dis_list.extend(cur_euclidean_list)
    # print('Euclidean Distance Number:', len(cur_euclidean_list))

    assert num_sample * (num_sample - 1) / 2 == len(cur_euclidean_list), 'Error'
    print("intra euclidean distance length:", len(intra_euclidean_dis_list))

inter_euclidean_dis_list = []
for i in tqdm.tqdm(range(class_number - 1)):
    class1_feature = np.squeeze(feature_test[i])
    num_img1 = class1_feature.shape[0]
    if num_img1 <= sample:
        num_sample1 = num_img1
        sample_feature1 = class1_feature
    else:
        num_sample1 = sample
        random_list1 = random.sample(list(range(num_img1)), num_sample1)
        sample_feature1 = class1_feature[random_list1]

    for j in range(i + 1, class_number):
        class2_feature = np.squeeze(feature_test[j])
        num_img2 = class2_feature.shape[0]
        if num_img2 <= sample:
            num_sample2 = num_img2
            sample_feature2 = class2_feature
        else:
            num_sample2 = sample
            random_list2 = random.sample(list(range(num_img2)), num_sample2)
            sample_feature2 = class2_feature[random_list2]

        Euclidean_dis = cdist(sample_feature1, sample_feature2, metric='euclidean')
        cur_euclidean_list = Euclidean_dis.flatten()
        inter_euclidean_dis_list.extend(cur_euclidean_list)

        print("inter euclidean distance length:", len(inter_euclidean_dis_list))

p = ranksums(intra_euclidean_dis_list, inter_euclidean_dis_list)
print(p.pvalue)

temp1 = intra_euclidean_dis_list + inter_euclidean_dis_list
temp2 = ['intra' for i in range(len(intra_euclidean_dis_list))] + ['inter' for i in range(len(inter_euclidean_dis_list))]
data_pic = pd.DataFrame({'euclidean distance':temp1,'tag':temp2})

fig,axes = plt.subplots(1,sharey=True)
sns.boxplot(x="tag", y="euclidean distance", data=data_pic, palette="Set3")

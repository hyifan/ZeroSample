# -- coding: utf-8 --
import cv2
import random
import operator
import numpy as np


def train_for_test_inputs(pathList, trainDict, labelsDict):
    res_images = []
    label_string = []
    label_number = []
    path = 'DataseB_20180919/train'
    for name in pathList:
        img = cv2.imread(path + '/' + name)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) # 去噪
        res_images.append(normalize(img))
        label_string.append(trainDict[name])
        label_number.append(labelsDict[trainDict[name]])
    return res_images, label_string, label_number

def get_test_trainNameSet(pathList, trainDict):
    realNameSet = set()
    f = open('DataseB_20180919/label_list.txt')
    for index, line in enumerate(f.readlines()):
        lists = line.replace('\n','').split('\t')
        realNameSet.add(lists[0])
    f.close()
    lists = []
    for name in pathList:
        lists.append(trainDict[name])
    set41 = set(lists)
    return realNameSet - set41

# 按类别包装训练数据，使随机选取的图片来自不同类别
def random_train_path(pathList, trainDict):
    res = {}
    for path in pathList:
        key = trainDict[path]
        if not res.has_key(key):
            res[key] = []
        res[key].append(path)
    return res

# 获取训练数据
def train_inputs(pathList, trainDict, labelsDict, num):
    path = 'DataseB_20180919/train'
    train_image = []
    train_label = []
    for labelId in random.sample(pathList, num):
        name = random.sample(pathList[labelId], 1)[0]
        img1 = cv2.imread(path + '/' + name) # 原图
        img1 = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21) # 去噪
        img2 = cv2.flip(img1, 1) # 水平翻转
        # img3 = cv2.resize(img1 ,(84, 84),interpolation=cv2.INTER_CUBIC)[10:74, 10:74] # 放大再裁剪
        # img4 = cv2.resize(img2 ,(84, 84),interpolation=cv2.INTER_CUBIC)[10:74, 10:74] # 水平翻转放大再裁剪
        # img5 = cv2.resize(img1 ,(84, 84),interpolation=cv2.INTER_CUBIC)[0:64, 10:74] # 放大再裁剪
        # img6 = cv2.resize(img2 ,(84, 84),interpolation=cv2.INTER_CUBIC)[0:64, 10:74] # 水平翻转放大再裁剪
        # img7 = cv2.resize(img1 ,(84, 84),interpolation=cv2.INTER_CUBIC)[10:74, 0:64] # 放大再裁剪
        # img8 = cv2.resize(img2 ,(84, 84),interpolation=cv2.INTER_CUBIC)[10:74, 0:64] # 水平翻转放大再裁剪

        # 调整亮度
        # img9 = np.uint8(np.clip((1.5 * img1 + 5), 0, 255))
        # img10 = np.uint8(np.clip((1.5 * img2 + 5), 0, 255))
        # img11 = np.uint8(np.clip((1.5 * img3 + 5), 0, 255))
        # img12 = np.uint8(np.clip((1.5 * img4 + 5), 0, 255))
        # img13 = np.uint8(np.clip((1.5 * img5 + 10), 0, 255))
        # img14 = np.uint8(np.clip((1.5 * img6 + 10), 0, 255))
        # img15 = np.uint8(np.clip((1.5 * img7 + 10), 0, 255))
        # img16 = np.uint8(np.clip((1.5 * img8 + 10), 0, 255))

        # images = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16]
        # images = [img1, img2, img3, img4, img5, img6, img7, img8]
        # for img in images:
        #     train_image.append(normalize(img))

        train_image.append(normalize(img1))
        train_image.append(normalize(img2))
        train_label.extend([labelsDict[trainDict[name]] for i in range(2)])
    return train_image, train_label


# def res_inputs2(pathList, step, batch_size):
#     path = 'DataseA_test/test'
#     images = []
#     names = []
#     start = step * batch_size
#     end = step * batch_size + batch_size
#     if end > 14633:
#         num = end - 14633
#         lists = pathList[start : end]
#         lists.extend(pathList[0 : num])
#     else:
#         lists = pathList[start : end]
#     for name in lists:
#         img = cv2.imread(path + '/' + name)
#         images.append(normalize(img))
#         names.append(name)
#     return images, names
# 获取测试数据
def res_inputs():
    images = []
    names = []
    path = 'DataseB_20180919/test'
    f = open('DataseB_20180919/image.txt')
    for index, line in enumerate(f.readlines()):
        name = line.replace('\n','')
        img = cv2.imread(path + '/' + name)
        # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) # 去噪
        images.append(normalize(img))
        names.append(name)
    f.close()
    return images, names


# 归一化
def normalize(image):
    return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# 获取训练图片对应的标签ID
def get_train_id():
    trainDict = {}
    trainNameList = []
    trainNameSet = set() # 分类类别
    f = open('DataseB_20180919/train.txt')
    for index, line in enumerate(f.readlines()):
        lists = line.replace('\n','').split('\t')
        trainDict[lists[0]] = lists[1]
        trainNameList.append(lists[0])
        trainNameSet.add(lists[1])
    f.close()
    return trainDict, trainNameList, trainNameSet


# 获取230个标签ID对应的30个特征
def get_labels_attr():
    labelsDict = {}
    labelsIdList = []
    labelsNameList = []
    f = open('DataseB_20180919/attributes_per_class.txt')
    for index, line in enumerate(f.readlines()):
        lists = line.replace('\n','').split('\t')
        attr = [float(i) for i in lists[1:]]
        labelsDict[lists[0]] = attr
        labelsIdList.append(attr)
        labelsNameList.append(lists[0])
    f.close()
    return labelsDict, labelsIdList, labelsNameList


# 获取语料信息
def get_word_attr(trainNameSet):
    realNameDict = {}
    f = open('DataseB_20180919/label_list.txt')
    for index, line in enumerate(f.readlines()):
        lists = line.replace('\n','').split('\t')
        realNameDict[lists[1]] = lists[0]
    f.close()
    wordNameList = []
    wordAttrList = []
    wordDict = {}
    f = open('DataseB_20180919/class_wordembeddings.txt')
    for index, line in enumerate(f.readlines()):
        lists = line.replace('\n','').split(' ')
        realName = realNameDict[lists[0]]
        attr = [float(i) for i in lists[1:]]
        wordDict[realName] = attr
        if realName not in trainNameSet:
            wordNameList.append(realName)
            wordAttrList.append(attr)
    f.close()
    return wordNameList, wordAttrList, wordDict

# def count_res(res):
#     for arr in res:
#         [max(i, 0.033) for i in arr]

# def to_labels_id_list(lists):
#     b = []
#     for line in lists:
#         sums = np.sum(line)
#         b.append([i/float(sums) for i in line])
#     return b

# 计算测试集所属类别
def count_res1(res, labelsIdList, labelsNameList):
    lists = []
    for arr in res:
        sumList = np.sum(np.abs(arr - np.array(labelsIdList)), axis=1)
        label = labelsNameList[np.argmin(sumList)]
        lists.append(label)
    return lists


# 计算不属于零样本类别的数据所属类别
def count_res2(res, wordNameList, wordAttrList, wordDict, trainNameSet):
    lists = []
    # i = 0
    for name in res:
        if name in trainNameSet:
            # i += 1
            sumList = np.sum(np.abs(np.array(wordDict[name]) - np.array(wordAttrList)), axis=1)
            label = wordNameList[np.argmin(sumList)]
            lists.append(label)
        else:
            lists.append(name)
    # print('test', i)
    return lists


def list_to_txt(res, res_names):
    f = open('submit.txt', 'w')
    for i in range(len(res)):
        f.write(res_names[i] + '\t' + res[i] + '\n')
    f.close()
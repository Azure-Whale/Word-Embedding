import numpy as np
import pickle
import matplotlib.pyplot as plt

##########  Import Word Lists  ###########################
f = open("./wordlists/male_pairs.txt")  # 返回一个文件对象
# line = f.readline()  # 调用文件的 readline()方法
Male = []
for line in f:
    # print(line, end = '')     # 在 Python 3 中使用
    # line = f.readline()
    line = line.strip('\n')
    Male.append(line)

f = open("./wordlists/female_pairs.txt")  # 返回一个文件对象
Female = []
for line in f:
    # print(line, end = '')     # 在 Python 3 中使用
    # line = f.readline()
    line = line.strip('\n')
    Female.append(line)
f.close()

f = open("./wordlists/occupations1950.txt")  # 返回一个文件对象
Occupation = []
for line in f:
    line = line.strip('\n')
    Occupation.append(line)
f.close()

# Occupation = list(set(Occupation))
# print(Male)
# print(Female)
# print(Occupation)
########################################################

'''
需要想个办法找出如何让python自动读取整个文件夹的内容而不用手动输入...
'''
year_lists = []
year_vec_lists = []
for i in range(80, 100):
    year_lists.append('./sgns/1' + str(i) + '0-vocab.pkl')
    year_vec_lists.append('./sgns/1' + str(i) + '0-w.npy')

'''
print(year_vec_lists,len(year_vec_lists))
x=input()
year_lists = ['./sgns/1800-vocab.pkl', './sgns/1810-vocab.pkl', './sgns/1820-vocab.pkl', './sgns/1830-vocab.pkl',
              './sgns/1840-vocab.pkl']
year_vec_lists = ['./sgns/1800-w.npy', './sgns/1810-w.npy', './sgns/1820-w.npy', './sgns/1830-w.npy',
                  './sgns/1840-w.npy']
'''


def Computation_Bias(year, year_vec):
    # get data sets of words
    with open(year, 'rb') as f:  # data_words are the vocabulary data set
        data_words = pickle.load(f)
    data_vectors = np.load(year_vec)  # data are the corresponding vectors

    #  Find corresponding vectors
    Group_male_vec = []
    for i in range(0, len(Male)):
        for j in range(0, len(data_words)):
            if Male[i] == data_words[j]:
                Group_male_vec.append(data_vectors[j])
    Group_male_vec = np.mat(Group_male_vec)  # total 20
    # print(Group_male_vec.shape)

    Group_female_vec = []
    for i in range(0, len(Female)):
        for j in range(0, len(data_words)):
            if Female[i] == data_words[j]:
                Group_female_vec.append(data_vectors[j])
    Group_female_vec = np.mat(Group_female_vec)  # total 20

    Occupation_vec = []
    for i in range(0, len(Occupation)):
        for j in range(0, len(data_words)):
            if Occupation[i] == data_words[j]:
                Occupation_vec.append(data_vectors[j])
    Occupation_vec = np.mat(Occupation_vec)  # total 152

    ### I2 Normalize vectors
    Group_male_vec = Group_male_vec / np.linalg.norm(Group_male_vec)
    Group_female_vec = Group_female_vec / np.linalg.norm(Group_female_vec)
    Occupation_vec = Occupation_vec / np.linalg.norm(Occupation_vec)

    #  Compute the average vectors for men and women group
    Group_man = Group_male_vec[0]
    for i in range(1, len(Group_male_vec)):
        Group_man += Group_male_vec[i]
    Group_man = Group_man / len(Group_male_vec)
    # print(Group_man.shape)
    Group_woman = Group_female_vec[0]
    for i in range(1, len(Group_female_vec)):
        Group_woman += Group_female_vec[i]
    Group_woman = Group_woman / len(Group_female_vec)
    # print(Group_female_vec[0,0])

    ### Bias Computation ###
    Bias = 0
    for i in range(0, len(Occupation_vec)):
        Bias += (np.linalg.norm(np.subtract(Occupation_vec[i], Group_woman))) - (np.linalg.norm(
            np.subtract(Occupation_vec[i], Group_man)))
    # print(Bias)
    return Bias


if __name__ == '__main__':
    with open('./sgns/1800-vocab.pkl', 'rb') as f:  # data2 are the vocabulary data set
        data2 = pickle.load(f)
    print(data2)
    print(Computation_Bias('./sgns/1800-vocab.pkl', './sgns/1800-w.npy'))

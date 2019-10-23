import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
from sklearn import linear_model
from sklearn import preprocessing
from random import randint
import gzip
from gensim.models import word2vec
from gensim.models import KeyedVectors
import logging

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# print(model['man'])

# x = input()

##########  Import Word Lists  ###########################
f = open("./wordlists/male_pairs.txt")  # return a file target
Male = []
for line in f:
    line = line.strip('\n')
    Male.append(line)
f.close()
'''
for i in Male:
    print(i)
    print(model[i])
    print(model[i].shape)
x=input()
'''
f = open("./wordlists/female_pairs.txt")  # return a file target
Female = []
for line in f:
    line = line.strip('\n')
    Female.append(line)
f.close()

f = open("./wordlists/occupations1950.txt")  # return a file target
Occupation = []
for line in f:
    line = line.strip('\n')
    Occupation.append(line)
f.close()

OccupationD = pd.read_csv("./wordlists/occupation_percentages_gender_occ1950.csv", header=0)  # return a file target
OccupationD = np.array(OccupationD)


#x=input()
'''
perc_90_20 = []
percentages = []
Occupation_difference = []
print(OccupationD[1][0])
for year in range(1910, 2000, 10):
    for i in range(0, len(OccupationD)):
        if year == OccupationD[i][0]:
            Occupation_difference.append(OccupationD[i][3])  # Gain All the percentages of occupation in selected year
    percentages = np.mean(Occupation_difference)  # get the average Percentage for woman in all occupations in seleted year
    perc_90_20.append(percentages)
    percentages = 0
Logit_Prop = []
for i in range(len(perc_90_20)):
    Logit_Prop.append(np.log(perc_90_20[i] / (1 - perc_90_20[i])))
'''
perc_90_20 = []
perc_90_20_std = []
percentages = []
Occupation_difference = []
print(OccupationD[1][0])
for year in range(1910, 2000, 10):
    for i in range(0, len(OccupationD)):
        if year == OccupationD[i][0]:
            Occupation_difference.append(
                100 * OccupationD[i][5])  # Gain All the percentages of occupation in selected year
    # print(year)
    # print(len(Occupation_difference))
    percentages = np.mean(
        Occupation_difference)  # get the average Percentage for woman in all occupations in seleted year
    percentages_std = np.std(Occupation_difference)
    Occupation_difference = []
    print(percentages)
    perc_90_20.append(percentages)
    perc_90_20_std.append(percentages_std)

combine = []
Occ_name = []
Dif = []  # calculate difference
# word lists of Occupation in 2015
for i in range(0, len(OccupationD)):
    if OccupationD[i][0] == 2015:
        Occ_name.append(OccupationD[i][1])
        Dif.append(100 * OccupationD[i][5])
        combine.append([Occ_name, Dif])
print(Occ_name, Dif)
print(len(Occ_name))

# x=input()

########################################################

'''

'''
year_lists = []
year_vec_lists = []
for i in range(91, 100):
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

Bias_Google = []
bk = 1


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
   # Group_male_vec = Group_male_vec / np.linalg.norm(Group_male_vec)
    #Group_female_vec = Group_female_vec / np.linalg.norm(Group_female_vec)
    #Occupation_vec = Occupation_vec / np.linalg.norm(Occupation_vec)

    #  Compute the average vectors for men and women group
    Group_man = Group_male_vec[0]
    for i in range(1, len(Group_male_vec)):
        Group_man += Group_male_vec[i]
    Group_man = Group_man / len(Group_male_vec)
    Group_woman = Group_female_vec[0]
    for i in range(1, len(Group_female_vec)):
        Group_woman += Group_female_vec[i]
    Group_woman = Group_woman / len(Group_female_vec)

    ### Bias Computation ###
    Bias = 0
    for i in range(0, len(Occupation_vec)):
        Bias += (np.linalg.norm(Occupation_vec[i] - Group_man)) - (np.linalg.norm(
            Occupation_vec[i] - Group_woman))
    Bias = Bias / len(Occupation_vec)

    return Bias


def Cal():

    Group_male_vec = []
    for item in Male:
        Group_male_vec.append(model[item])
    Group_male_vec = np.array(Group_male_vec)  # total 20
    #print(Group_male_vec.shape)
    #Group_male_vec = Group_male_vec / np.linalg.norm(Group_male_vec)

    Group_female_vec = []
    for item in Female:
        Group_female_vec.append(model[item])
    Group_female_vec = np.array(Group_female_vec)  # total 20
   # print('woman vectors')
    #print(Group_female_vec)
    #Group_female_vec = Group_female_vec / np.linalg.norm(Group_female_vec)

    Occupation_vec_2015 = []
    list_index = []
    for item in Occ_name:
        Occupation_vec_2015.append(model[item])
    Occupation_vec_2015 = np.array(Occupation_vec_2015)  # total 152
    #Occupation_vec_2015 = Occupation_vec_2015 / np.linalg.norm(Occupation_vec_2015)
    #x = input()
    '''
    for i in range(0, len(Male)):
        for j in range(0, len(data_words)):
            if Male[i] == data_words[j]:
                Group_male_vec.append(data_vectors[j])
    print(Group_male_vec)
    Group_male_vec = np.mat(Group_male_vec)  # total 20
    print(Group_male_vec)

    Group_female_vec = []
    for i in range(0, len(Female)):
        for j in range(0, len(data_words)):
            if Female[i] == data_words[j]:
                Group_female_vec.append(data_vectors[j])
    Group_female_vec = np.mat(Group_female_vec)  # total 20

    # print(Occ_name)
    Occupation_vec_2015 = []
    list_index = []
    for i in range(0, len(Occupation)):
        for j in range(0, len(data_words)):
            if Occ_name[i][0] == data_words[j]:
                list_index.append(i)
                Occupation_vec_2015.append(data_vectors[j])
    Occupation_vec_2015 = np.mat(Occupation_vec_2015)  # total 152
    '''
    ### I2 Normalize vectors
    # Group_male_vec = Group_male_vec / np.linalg.norm(Group_male_vec)
    # Group_female_vec = Group_female_vec / np.linalg.norm(Group_female_vec)
    # Occupation_vec_2015 = Occupation_vec_2015 / np.linalg.norm(Occupation_vec_2015)

    #  Compute the average vectors for men and women group
    Group1 = np.mean(np.array(Group_male_vec), axis=0)
    Group2 = np.mean(np.array(Group_female_vec), axis=0)
    '''
    print('test+', Group1.shape)
    Group_man = Group_male_vec[0]
    for i in range(1, len(Group_male_vec)):
        Group_man += Group_male_vec[i]
    Group_man = Group_man / len(Group_male_vec)
    '''

    '''
    Group_woman = Group_female_vec[0]
    for i in range(1, len(Group_female_vec)):
        Group_woman += Group_female_vec[i]
    Group_woman = Group_woman / len(Group_female_vec)
    '''
    for i in range(0, len(Occupation_vec_2015)):
        Bias_Google.append((np.linalg.norm(np.subtract(Occupation_vec_2015[i], Group1))) - (np.linalg.norm(
            np.subtract(Occupation_vec_2015[i], Group2))))
    '''
    difference = []
    for i in list_index:
        difference.append(Dif[i])
    for i in range(0, len(difference)):
        print(Bias_Google[i], difference[i])
    '''
    sns.regplot(x=Dif, y=Bias_Google)
    plt.scatter(Dif, Bias_Google, alpha=0.6)

    # Figure 1
    plt.xlim([-100, 100])
    plt.xlabel('Woman Occupation(%) Difference')
    plt.ylabel('Woman Bias')

    plt.show()
    #print(len(difference))
    print(len(list_index))
    print(len(Bias_Google))


if __name__ == '__main__':
    year = []
    for i in range(1910, 2000, 10):
        year.append(i)
    Cal()
    Bias_All = []
    for a, b in zip(year_lists, year_vec_lists):  # zip and traverse
        Bias_All.append(Computation_Bias(a, b))
    # print(len(Bias_All))
    # print(Bias_All)

    fig = plt.figure()
    plt.grid()
    ax = fig.add_subplot(111)
    ax.plot(year, Bias_All, color='blue', label='Bias')
    ax.legend(loc=1)
    plt.xlabel('Year')
    ax.set_ylabel('Bias')
    ax2 = ax.twinx()
    ax2.plot(year, perc_90_20, color='red', label='Difference')
    # ax2.set_ylabel('Women Occupation Logit Prop')
    ax2.set_ylabel('Difference(%) of woman occupation')
    ax2.legend(loc=2)
    print(len(perc_90_20))
    print(year[0])
    print(perc_90_20_std[0])
    # for i in range(0,9):
    # ax2.fill_between(year[i], perc_90_20[i] - percentages_std[i], perc_90_20[i] + perc_90_20_std[i], alpha=0.35)
    plt.xticks(year)
    plt.xlim([1910, 1990])
    plt.show()

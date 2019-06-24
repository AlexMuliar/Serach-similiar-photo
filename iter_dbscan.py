import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.neighbors import LSHForest, KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from PIL import Image

from matplotlib.pyplot import imshow

save_to = sys.argv[1]#'./result_data/27'
take_of = sys.argv[2]#'test_data'
os.system(f'rm -rf { save_to }; mkdir { save_to }')
log_file  = ''

def get_photo(path, size):
    photos = dict()
    names = os.listdir(f'./{ path }')

    for name in names:
        img = cv2.imread(f'./{ path }/{ name }')
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            continue
        img = cv2.resize(img, size)

        photos.update({name: img})
    print(len(photos))
    return photos

def search_distance(data):
    #num_clusters = list()
    indef = list()
    iteration = 0
    while True:
        iteration += 1
        clf = DBSCAN(eps=iteration, min_samples=3).fit([frame for frame in data])
        n = len(np.unique(clf.labels_))
        #num_clusters.append(n)
        n_clusters = [len(np.where(clf.labels_ == i)[0]) for i in range(n)]
        indef.append(len(np.where(clf.labels_ == -1)[0]) + max(n_clusters))
        if not -1 in clf.labels_:
            break
    dist = np.where(np.array(indef) == min(indef))[0][0] + 1
    save_plot(indef)
    write_to_log(f'Count of noise: { min(indef) }\nDistance: { dist }:\n')
    return dist

def get_clusters(data):
    clf = DBSCAN(eps=search_distance(data), min_samples=3).fit([frame for frame in data])
    clusters = list()
    for i in range(-1, len(np.unique(clf.labels_))):
        clusters.append(np.where(clf.labels_ == i)[0])
    return clusters

def get_noise(clusters):
    count = [len(i) for i in clusters]
    if len(count) == 1:
        return True
    elif len(count) == 2:
        return []
    else:
        return [0, count.index(max(count[1:]))]

    
def save_cluster(folder, data, photos):
    names = list(photos.keys())
    if folder == 'noise':
        for i in data:
            cv2.imwrite(f'./{ save_to }/noise/{ names[i] }', cv2.imread(f'./{ take_of }/{ names[i] }'))
    if folder == 'data':
        index = 0
        try:
            index = max([int(j) for j in os.listdir(f'./{ save_to }/data')]) + 1
            #print("list of dirs:" + os.listdir(f'./{ save_to }/data'))
        except Exception as ex:
            pass
        finally:
            os.system(f'mkdir ./{ save_to }/data/{ index }')
        for i in data:
            cv2.imwrite(f'./{ save_to }/data/{ index }/{ names[i] }', cv2.imread(f'./{ take_of }/{ names[i] }'))
            cv2.imwrite(f'./{ save_to }/all/{ names[i] }', cv2.imread(f'./{ take_of }/{ names[i] }'))


def save_plot(indef):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xlabel('Distanse')
    plt.ylabel('Noise')
    plt.plot(range(1, len(indef) + 1), indef)
    plt.savefig(f'{ save_to }/log/{ min(indef) }.png', dpi=200)
    
def write_to_log(data):
    file = open(f'{save_to}/log/{ log_file }.txt', 'a+')
    file.write(str(data) + '\n')
    file.close()
    
def write_log(clusters, noise):
    file = open(f'{ save_to }/log/{ log_file }.txt', 'a+')
    data = '\n'
    for i in range(len(clusters)):
        data += f'#{i}({len(clusters[i])}): {clusters[i]}'
        if i in noise:
            data += '- NOISE' 
        data += '\n'
    file.write(data)
    file.close()

def iterator(n_iter):
    global log_file
    log_file = f'log_{ datetime.datetime.now() }'
    os.system(f'mkdir ./{ save_to }/noise; mkdir ./{ save_to }/data;  mkdir ./{ save_to }/log; mkdir ./{ save_to }/all;')
    size = 16, 16
    photos = get_photo(f'./{ take_of }', size)
    for itr in range(n_iter):
        data = [i.reshape(1, size[0] * size[1])[0] for i in photos.values()]
        #print(len(photos), len(data))
        if not data:
            break
        clusters = get_clusters(data)
        #print(sum(len(i) for i in clusters))
        noise = get_noise(clusters)
        if noise == True:
            save_cluster('noise', clusters[0], photos)
            write_log(clusters, [])
            break
        elif noise == []:
            save_cluster('noise', clusters[0], photos)
            save_cluster('noise', clusters[1], photos)
            write_log(clusters, noise)
            break
        write_log(clusters, noise)
        print(f'{itr}: Finded { len(clusters) - len(noise) - 1 }')
        for i in range(len(clusters) - 1):
            if i in noise:
                save_cluster('noise', clusters[i], photos)
            else:
                save_cluster('data', clusters[i], photos)
        photos = get_photo(f'./{ save_to }/noise', size)
        
        os.system(f'rm -rf ./{save_to}/noise; mkdir ./{ save_to }/noise;')
#         if itr % 4 == 0 and itr != 0:
#             size = (size[0] * 2, size[1] * 2)

if __name__ == "__main__":
    iterator(20)


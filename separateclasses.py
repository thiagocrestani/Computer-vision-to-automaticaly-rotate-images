# -*- coding: utf-8 -*-

import sys
import csv
import shutil
import os


def load_csv(fn):
    res = {}
    for row in csv.DictReader(open(fn)):
        res[row['fn']] = row['label']
    return res

truth = load_csv('train.rotfaces/train.truth.csv')


i = 0
if not os.path.exists('dataset'):
    os.mkdir('dataset')
    os.mkdir('dataset/train')
    os.mkdir('dataset/test')  
for image in truth:
    print(image)
    if(i <= 7):
        if not os.path.exists('dataset/train/'+truth[image]):
            os.mkdir('dataset/train/'+truth[image])    
        shutil.copy2('train.rotfaces/train/'+image, 'dataset/train/'+truth[image]+'/'+image)
    else:
        if not os.path.exists('dataset/test/'+truth[image]):
            os.mkdir('dataset/test/'+truth[image])    
        shutil.copy2('train.rotfaces/train/'+image, 'dataset/test/'+truth[image]+'/'+image)
    i += 1
    if i == 10:
        i = 0
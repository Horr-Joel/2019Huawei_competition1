import pandas as pd
import numpy as np
import config
import gc
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict as dd
from keras.utils import to_categorical

class DataLoader():
    def __init__(self,chunksize, **kwargs):
        self.chunksize = chunksize

        feature_dict = {}

        print('load behavior info...')
        behavior = pd.read_csv('../data/user_behavior_info.csv',header=None)
        behavior.columns = ['adId','billId','primId','creativeType','intertype','spreadAppId']
        ad_info.fillna(-1,inplace=True)
        for f in ['adId','primId','creativeType','intertype','spreadAppId']:
            ad_info[f] = ad_info[f].astype(np.int32)

        print('load app info...')
        app_info = pd.read_csv('../data/app_info.csv',header=None)
        app_info.columns = ['uId','age','gender','city','province','phoneType','carrier']
        for f in ['age','gender','city','province','phoneType','carrier']:
            user_info[f] = user_info[f].astype(np.float32)
        user_info.fillna(-1,inplace=True)

        print('load user info...')
        user_info = pd.read_csv('../data/user_basic_info.csv',header=None)
        user_info.columns=['contentId','firstClass','secondClass']
        user_info['contentId'] = content_info['contentId'].astype(np.int32)
        #content_info['secondClass'] = content_info['secondClass'].apply(lambda x: x.split('#') if isinstance(x,str) else [])
        content_info.fillna('Nan',inplace=True)
        content_info['firstClass'] = LabelEncoder().fit_transform(content_info['firstClass'])
        # remain be improved
        content_info['secondClass'] = LabelEncoder().fit_transform(content_info['secondClass'])


        print('load actived_app info...')
        actived_app = pd.read_csv('../data/user_app_actived.csv',header=None)



        self.data = pd.read_csv('../data/age_train.csv',header=None)
        self.data.columns = ['uId','age_group']

        self.train_length = self.data.shape[0]
        test = pd.read_csv('../data/age_test.csv',header=None)
        test.columns = ['uId']
        self.data = self.data.append(test)


        self.data = self.data.merge(user_info,how='left',on='uId')
        self.data = self.data.merge(behavior,how='left',on='uId')
        self.data = self.data.merge(actived_app,how='left',on='uId')

        for f in config.CATEGORECIAL_COLS:
            self.data[f] = LabelEncoder().fit_transform(self.data[f])
            feature_dict[f] = self.data[f].nunique()


        self.feature_dict = feature_dict

        del feature_dict
        del user_info
        del behavior
        del actived_app
        gc.collect()


    def get_feature_dict(self):
        return self.feature_dict

    def get_feature_le(self):
        return self.le_dict

    def get_train(self):
        train = self.data.iloc[self.train_length:]
        label = train['age_group']-1
        del train['age_group']
        label = to_categorical(label)
        return train, label

    def get_test(self):
        test = self.data.iloc[:self.train_length]
        del test['age_group']
        return test
    
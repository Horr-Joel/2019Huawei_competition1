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
    def __init__(self, **kwargs):
        if os.path.exists('../data/data.csv'):
            print('start load data...')
            self.data = pd.read_csv('../data/data.csv')
        else:
            print('load behavior info...')
            behavior = pd.read_csv("../data/user_behavior_info.csv",header=None)
            behavior.columns = ['uId','bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','GFuncTimes']

            behavior['ABCDEFTimes'] = behavior['AFuncTimes'] + behavior['BFuncTimes'] + behavior['CFuncTimes'] + behavior['DFuncTimes'] + \
                                      behavior['EFuncTimes'] + behavior['FFuncTimes']
            behavior['G_boot'] = behavior['GFuncTimes'] / behavior['bootTimes']
            behavior['A_all'] = behavior['AFuncTimes'] / behavior['ABCDEFTimes']
            behavior['B_all'] = behavior['BFuncTimes'] / behavior['ABCDEFTimes']
            behavior['C_all'] = behavior['CFuncTimes'] / behavior['ABCDEFTimes']
            behavior['D_all'] = behavior['DFuncTimes'] / behavior['ABCDEFTimes']
            behavior['E_all'] = behavior['EFuncTimes'] / behavior['ABCDEFTimes']
            behavior['F_all'] = behavior['FFuncTimes'] / behavior['ABCDEFTimes']
            behavior['all_boot'] = behavior['ABCDEFTimes']/ behavior['bootTimes']


            print('load app info...')
            app_info = pd.read_csv("../data/app_info.csv",header=None)
            app_info.columns = ['appId','category']


            print('load user info...')
            user_info = pd.read_csv('../data/user_basic_info.csv',header=None)
            user_info.columns = ['uId','gender','city','prodName','ramCapacity','ramLeftRation','romCapacity'
            ,'romLeftRation','color','fontSize','ct','carrier','os']

            user_info['ramLeft'] = user_info['ramCapacity'] * user_info['ramLeftRation']
            user_info['romLeft'] = user_info['romCapacity'] * user_info['romLeftRation']
            user_info['rom_ram'] = user_info['romCapacity'] / user_info['ramCapacity']
            user_info['ct_2g'] = user_info['ct'].apply(lambda x: 1 if isinstance(x,str) and '2g' in x else 0)
            user_info['ct_3g'] = user_info['ct'].apply(lambda x: 1 if isinstance(x,str) and '3g' in x else 0)
            user_info['ct_4g'] = user_info['ct'].apply(lambda x: 1 if isinstance(x,str) and '4g' in x else 0)
            user_info['ct_wifi'] = user_info['ct'].apply(lambda x: 1 if isinstance(x,str) and 'wifi' in x else 0)
            del user_info['ct']

            user_info['os_first'] = user_info['os'].apply(lambda x:int(x) if not np.isnan(x) else -1)


            print('load actived_app info...')
            active = pd.read_csv("data/user_app_actived.csv",header=None)
            active.columns = ['uId','appId']

            active['appId'] = active['appId'].apply(lambda x:x.split('#'))
            active['appNum'] = active['appId'].apply(lambda x:len(x) if x[0]!='\\N' else 0)


            key = app_info.appId.values
            val = app_info.category.values
            app_map = dd(int)
            for i in range(len(key)):
                 app_map[key[i]] = val[i]

            def app_cate_data(x,t):
                cate = dd(int)
                for each in x:    
                    cate[app_map[each]] += 1
                tmp = cate.values()
                s = sum(tmp)+1
                # all_num
                if t == 0:
                    return len(cate)
                # max_num
                elif t == 1:
                    return max(tmp)/s
                # min_num
                else:
                    return min(tmp)/s
                    
            active['app_cate_num'] = active['appId'].apply(lambda x: app_cate_data(x,0))
            active['app_cate_mean'] = active['appNum']/active['app_cate_num']
            active['app_cate_maxRate'] = active['appId'].apply(lambda x: app_cate_data(x,1))
            active['app_cate_minRate'] = active['appId'].apply(lambda x: app_cate_data(x,2))
            active['app_cate_max'] = active['app_cate_maxRate'] * (active['appNum']+1)
            active['app_cate_min'] = active['app_cate_minRate'] * (active['appNum']+1)
            del active['appId']

            print('merge data..')
            self.data = pd.read_csv('../data/age_train.csv',header=None)
            self.data.columns = ['uId','age_group']

            self.train_length = self.data.shape[0]
            test = pd.read_csv('../data/age_test.csv',header=None)
            test.columns = ['uId']
            self.data = self.data.append(test)


            self.data = self.data.merge(user_info,how='left',on='uId')
            self.data = self.data.merge(behavior,how='left',on='uId')
            self.data = self.data.merge(active,how='left',on='uId')


            for f1 in ['prodName','city','gender','color','os_first','carrier','ct_2g','ct_3g','ct_4g','ct_wifi']:
                # Numerical feature
                for f2 in ['appNum','ramCapacity','romCapacity','fontSize','bootTimes','AFuncTimes','BFuncTimes',
                   'CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','GFuncTimes']:
                    self.data = self.data.merge(data.groupby(f1)[f2].agg(['mean','min','max','std','size']).reset_index(),how='left',on=f1)
                # category feature


            self.data.to_csv('../data/data.csv',index=False)
            del user_info
            del behavior
            del active

        self.data.fillna(-1,inplace=True)
        feature_dict = {}
        for f in config.CATEGORECIAL_COLS:
            self.data[f] = LabelEncoder().fit_transform(self.data[f])
            feature_dict[f] = self.data[f].nunique()
        self.feature_dict = feature_dict

        del feature_dict
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
    
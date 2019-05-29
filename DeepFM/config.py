

#set the path to files
TRAIN_FILE = '../data/age_train.csv'
TEST_FILE = '../data/age_test.csv'

MODEL_FILE = './model/model.h5'
SUB_DIR = './output'

RANDOMSTATE = 2019

EPOCH = 1
BATCH_SIZE = 2048
EMBEDDING_SIZE = 8

TRAIN_BATCH_SIZE = 2010000

CATEGORECIAL_COLS =[
     'gender', 'city', 'prodName',  'color', 'ct', 'carrier', 'os'
]

NUMERIC_COLS = [
    'ramCapacity', 'ramLeftRation', 'romCapacity', 'romLeftRation','fontSize','bootTimes', 'AFuncTimes', 'BFuncTimes',
    'CFuncTimes', 'DFuncTimes', 'EFuncTimes', 'FFuncTimes', 'GFuncTimes','appNum','app_cate_num',
    'app_cate_mean','app_cate_maxRate','app_cate_minRate','app_cate_max','app_cate_min'
]

IGNORE_COLS = [
    'uId'
]

from deepfm import KerasDeepFM
from DataLoader import DataLoader
import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc
from keras.models import load_model
from mylayers import MySumLayer
from metrics import auc
if __name__ == "__main__":
    has_model = True
    dl = DataLoader()
    if has_model:
        kfm = load_model(config.MODEL_FILE, custom_objects={'MySumLayer':MySumLayer,'auc':auc})
    else:
        print("starting read data and preprocessing data...")
        feat_dict = dl.get_feature_dict()
        kfm = KerasDeepFM(config.EMBEDDING_SIZE, feat_dict)



        train = dl.get_train()
        x_train,x_val,y_train,y_val = train_test_split(train[config.NUMERIC_COLS+config.CATEGORECIAL_COLS],batch['age_group'],test_size=0.2,random_state=config.RANDOMSTATE)
        x_train = x_train.values.T
        x_train = [np.array(x_train[i,:]) for i in range(x_train.shape[0])]
        y_train = y_train.values

        x_val = x_val.values.T
        x_val = [np.array(x_val[i,:]) for i in range(x_val.shape[0])]
        y_val = y_val.values

        print('starting train...')
        kfm.fit(x_train, y_train, x_val, y_val,config.EPOCH,config.BATCH_SIZE)

        kfm.save()


    if 1:
        print('start load testset...')
        test = dl.get_test()
        test = test[config.NUMERIC_COLS+config.CATEGORECIAL_COLS]
        test = test.values.T
        test = [np.array(test[i,:]) for i in range(test.shape[0])]

        print('start predict testset...')
        predict = kfm.predict(test)

        sub = test[test['uId']]
        sub.columns = ['id']
        sub['label'] = predict
        sub.to_csv('../output/submission.csv',index=False)




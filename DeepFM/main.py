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
    dl = DataLoader(config.TRAIN_BATCH_SIZE)
    if has_model:
        kfm = load_model(config.MODEL_FILE, custom_objects={'MySumLayer':MySumLayer,'auc':auc})
    else:
        print("starting read data and preprocessing data...")
        feat_dict = dl.get_feature_dict()
        kfm = KerasDeepFM(config.EMBEDDING_SIZE, feat_dict)

        loop = True
        i = 0
        maxturn = 2
        while loop and i < maxturn:
            try:
                print('starting load No.%d Batch...' % i)
                batch = dl.get_next()
                x_train,x_val,y_train,y_val = train_test_split(batch[config.NUMERIC_COLS+config.CATEGORECIAL_COLS],batch['label'],test_size=0.2,random_state=config.RANDOMSTATE)
                x_train = x_train.values.T
                x_train = [np.array(x_train[i,:]) for i in range(x_train.shape[0])]
                y_train = y_train.values

                x_val = x_val.values.T
                x_val = [np.array(x_val[i,:]) for i in range(x_val.shape[0])]
                y_val = y_val.values

                print('starting train No.%d Batch...' % i)
                kfm.fit(x_train, y_train, x_val, y_val,config.EPOCH,config.BATCH_SIZE)
                i += 1
                kfm.save()
            except StopIteration:
                loop = False
                print("training is stopped.")

    if 1:
        print('start load testset...')
        test = dl.get_test()
        sub = pd.DataFrame(test['label'].values,columns=['id'])
        test = test[config.NUMERIC_COLS+config.CATEGORECIAL_COLS]
        test = test.values.T
        test = [np.array(test[i,:]) for i in range(test.shape[0])]

        print('start predict testset...')
        predict = kfm.predict(test)

        sub['probability'] = predict
        sub.to_csv('../output/submission.csv',index=False)




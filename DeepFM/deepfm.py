import numpy as np
from keras.layers import Input, Dense, Embedding, Add, Concatenate, RepeatVector,Multiply,Subtract,Lambda,Dropout,Reshape,Flatten
from keras.models import Model
from keras.utils import plot_model
from mylayers import MySumLayer
from keras.optimizers import Adam
import config
from keras.metrics import binary_accuracy
import warnings

warnings.filterwarnings('ignore')



class KerasDeepFM(object):
    def __init__(self,k, feat_dict):
        # k is the number of embedding dim
        # feat_dict is the unique number of every features
        input_cols = []
        embed_col = []
        if config.NUMERIC_COLS:
            #numeric cols
            numeric_cols = []
            for col in config.NUMERIC_COLS:
                in_neu = Input(shape=(1,), name=col)            #None*1
                input_cols.append(in_neu)
                in_embed = RepeatVector(1)(Dense(k)(in_neu))    #None*1*k
                numeric_cols.append(in_neu)
                embed_col.append(in_embed)

            con_numeric = Concatenate(axis=1)(numeric_cols)        #None*len(config.NUMERIC_COLS)
            dense_numeric = RepeatVector(1)(Dense(1)(con_numeric))    #None*1*1

        #categorical cols
        categorical_cols = []
        for col in config.CATEGORECIAL_COLS:
            in_cate = Input(shape=(1,),name=col)            #None*1
            input_cols.append(in_cate)
            cate_embedding = Embedding(feat_dict[col], 1)(in_cate)    #None*1*1
            in_embed = Embedding(feat_dict[col], k)(in_cate)        #None*1*k
            embed_col.append(in_embed)
            categorical_cols.append(cate_embedding)
        con_cate = Concatenate(axis=1)(categorical_cols)        #None*len(config.CATEGORECIAL_COLS)*1
        #first order
        if config.NUMERIC_COLS:
            y_first_order = Concatenate(axis=1)([dense_numeric, con_cate])         #None*len*1
        else:
            y_first_order = con_cate
            
        y_first_order = MySumLayer(axis=1)(y_first_order)                #None*1

        #second order
        emb = Concatenate(axis=1)(embed_col)                        #None*s*k

        summed_feature_emb = MySumLayer(axis=1)(emb)                #None*k
        summed_feature_emb_squred = Multiply()([summed_feature_emb,summed_feature_emb])    #None*k

        squared_feature_emb = Multiply()([emb,emb])                    #None*s*k
        squared_sum_feature_emb = MySumLayer(axis=1)(squared_feature_emb)    #None*k

        sub = Subtract()([summed_feature_emb_squred,squared_sum_feature_emb])    #None*k
        sub = Lambda(lambda x: x*0.5)(sub)                        #None*k
        y_second_order = MySumLayer(axis=1)(sub)                    #None*1

        #deep order
        y_deep = Flatten()(emb)                                #None*(s*k)
        y_deep = Dropout(0.5)(Dense(32,activation='relu')(y_deep))            #None*32
        y_deep = Dropout(0.5)(Dense(32,activation='relu')(y_deep))            #None*32
        y_deep = Dropout(0.5)(Dense(1,activation='relu')(y_deep))            #None*1

        #deep fm
        y = Concatenate()([y_first_order,y_second_order,y_deep])            #None*3
        y = Dense(6,activation='softmax')(y)                        #None*1
        self.model = Model(inputs=input_cols, outputs=[y])
        # self.model.summary()
        self.model.compile(optimizer=Adam(lr=0.01,decay=0.1), loss='categorical_crossentropy',metrics=['accuracy'])

    def fit(self, x_train, y_train,x_val,y_val,epochs,batch_size):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val,y_val))

    def predict(self,x):
        y_pred = self.model.predict(x)
        return y_pred

    def save(self):
        self.model.save(config.MODEL_FILE)




import os
from config import use_device
os.environ["CUDA_VISIBLE_DEVICES"] = use_device
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
from keras import backend as K
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint,Callback,LearningRateScheduler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import log_loss
import datetime

def lr_de(epoch,lr):
    if epoch==0:
        return lr
    elif lr>0.0002:
            return lr/2
    else:
        return lr

class epochHistory(Callback):

    def on_train_begin(self, logs=None):
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)

def iter_ense(epochs,model,te):

    result = 0
    for e in epochs[-3:]:
        model.load_weights('./weight/weights.'+str(e+1)+'.hdf5')
        result += model.predict(te, batch_size=1024)
    return result/3


def train(use_data,semi_sv,output,data_aug,use_model):

    def get_subset(dataset,idx):
        data = {}
        for key,value in dataset.items():
            data[key] = value[idx]
        return data

    def concat_data(data1,data2):
        result = {}
        for k in data1.keys():
            result[k] = np.concatenate([data1[k],data2[k]])
        return result

    def get_aug_data(tr_x, tr_y):
        tr_q1 = tr_x['q1']
        tr_q2 = tr_x['q2']
        tr_gf = tr_x['gf']
        tr_q1node = tr_x['q1node']
        tr_q2node = tr_x['q2node']

        res_q1 = []
        res_q2 = []
        res_gf = []
        res_q1node = []
        res_q2node = []
        res_y = []

        for q1, q2, gf, q1node, q2node, y in zip(tr_q1, tr_q2, tr_gf, tr_q1node, tr_q2node, tr_y):
            r1 = q1[np.in1d(q1, q2, invert=True)]
            len1 = len(r1)
            if len1 < 4 or len1==len(q1[q1!=0]):
                continue

            r2 = q2[np.in1d(q2, q1, invert=True)]
            len2 = len(r2)
            if len2 < 4 or len2==len(q2[q2!=0]):
                continue

            out1 = np.zeros(15, dtype=np.int32)
            out2 = np.zeros(15, dtype=np.int32)
            out1[-len1:] = r1
            out2[-len2:] = r2

            res_q1.append(out1)
            res_q2.append(out2)
            res_gf.append(gf)
            res_q1node.append(q1node)
            res_q2node.append(q2node)
            res_y.append(y)


        res_x = {
            'q1': np.asarray(res_q1),
            'q2': np.asarray(res_q2),
            'gf': np.asarray(res_gf),
            'q1node': np.asarray(res_q1node),
            'q2node': np.asarray(res_q2node)
        }
        res_y = np.asarray(res_y)
        return res_x, res_y

    from nn import rnnword, aggmodel, esim,attention,rnn_res
    if use_model == 'rnnword':
        get_model = rnnword
    elif use_model == 'aggmodel':
        pass
    elif use_model == 'esim':
        get_model = esim
    elif use_model == 'attention':
        get_model = attention
    elif use_model == 'res':
        get_model = rnn_res
    else:
        raise RuntimeError("don't have this model")

    from readdata import read_data

    model_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_'+use_data+'_'+str(semi_sv)+'_'+str(data_aug)+'_'

    tr,te, embedding_matrix, labels = read_data(use_data,data_aug=data_aug)

    print(use_data)
    print('Shape of label tensor:', labels.shape)

    y = labels

    from config import model_path
    from sklearn.cross_validation import StratifiedKFold, KFold
    from config import n_folds

    y_pred = pd.read_csv("./data/y_pred.csv")['y_pre'].values
    y_pos_ = y_pred == 1
    y_neg_ = y_pred == 0
    add_idx = np.any([y_pos_, y_neg_], axis=0)
    add_y = y_pred[add_idx]


    y_pos = y_pred > 0.75
    y_neg = y_pred < 0.25
    y_idx = np.any([y_pos, y_neg], axis=0)
    y_pred = y_pred[y_idx]
    print(y_idx.shape)


    folds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
    result = np.zeros((len(te['q1']), 1))

    oof_y = np.zeros((len(y), 1))
    for n_fold, (tr_idx, val_idx) in enumerate(folds):
        tr_x = get_subset(tr,tr_idx)
        tr_y = y[tr_idx]
        # if data_aug:
        #     res_x,res_y = get_aug_data(tr_x,tr_y)
        #     tr_x = concat_data(tr_x,res_x)
        #     tr_y = np.concatenate([tr_y,res_y])

        if semi_sv:
            te_x = get_subset(te, y_idx)
            tr_data = concat_data(tr_x,te_x)
            tr_y = np.concatenate([tr_y,y_pred])
            patience = 3
        else:
            add_data = get_subset(te,add_idx)
            tr_data = concat_data(tr_x,add_data)
            tr_y = np.concatenate([tr_y, add_y])
            patience = 2
            # tr_data = tr_x
            # tr_y = y[tr_idx]

        val_x = get_subset(tr, val_idx)
        val_y = y[val_idx]

        use_word = True
        if use_data!='words':
            use_word = False
        model = get_model(word_embedding_matrix=embedding_matrix,use_word=use_word)
        if n_fold == 0:
            print(model.summary())

        # hist = epochHistory()
        print(n_fold)
        model.fit(tr_data,
                  tr_y,
                  epochs=1000,
                  validation_data=[val_x,val_y],
                  verbose=1,
                  batch_size=256,
                  callbacks=[
                      EarlyStopping(patience=patience, monitor='val_binary_crossentropy'),
                      # LearningRateScheduler(lr_de,verbose=1)
                      # hist,
                      # ModelCheckpoint('./weight/weights.{epoch:d}.hdf5',monitor='val_binary_crossentropy',save_weights_only=True)
                  ])
        # result += iter_ense(hist.epochs,model,te)
        result += model.predict(te, batch_size=1024)

        model.save_weights('./weight/'+model_name+str(n_fold)+'.h5')
        # oof_y[val_idx] = model.predict(val_x, batch_size=2048)

        K.clear_session()
        tf.reset_default_graph()

    # 提交结果
    result /= n_folds
    submit = pd.DataFrame()
    submit['y_pre'] = list(result[:, 0])
    submit.to_csv(output, index=False)


    ## 保存预测的训练标签
    # oof_y = oof_y[:,0]
    # oof_y_ = oof_y.round().astype(int)
    #
    # error_idx = oof_y_!=y
    # print(np.sum(error_idx))
    # oof_y[error_idx] = 1-oof_y[error_idx]
    submit = pd.DataFrame()
    submit['y_pre'] = oof_y[:,0]
    submit.to_csv('./data/oofy.csv',index=False)


"""
train('words',False,'esim_word0_2.csv',False,'esim')
train('words',True,'esim_word1_2.csv',False,'esim')
train('chars',False,'esim_char0_2.csv',False,'esim')
train('chars',True,'esim_char1_2.csv',False,'esim')

train('words',False,'esim_word0_3.csv',False,'esim')
train('words',True,'esim_word1_3.csv',False,'esim')
train('chars',False,'esim_char0_3.csv',False,'esim')
train('chars',True,'esim_char1_3.csv',False,'esim')

train('words',False,'attention_word0_0.csv',False,'attention')
train('chars',True,'attention_char1_0.csv',False,'attention')

train('words',True,'attention_word1_0.csv',False,'attention')
train('chars',False,'attention_char0_0.csv',False,'attention')
"""

"""
train('words',False,'attention_word0_1.csv',False,'attention')
train('chars',True,'attention_char1_1.csv',False,'attention')

train('words',True,'attention_word1_1.csv',False,'attention')
train('chars',False,'attention_char0_1.csv',False,'attention')

"""

"""
train('words',False,'attention_word0_2.csv',False,'attention')
train('chars',True,'attention_char1_2.csv',False,'attention')

train('words',True,'attention_word1_2.csv',False,'attention')
train('chars',False,'attention_char0_2.csv',False,'attention')
"""

"""
train('words',False,'attention_word0_3.csv',False,'attention')
train('chars',True,'attention_char1_3.csv',False,'attention')

train('words',True,'attention_word1_3.csv',False,'attention')
train('chars',False,'attention_char0_3.csv',False,'attention')
"""


train('words',False,'res_word0_2.csv',False,'res')
train('chars',True,'res_char1_2.csv',False,'res')

train('words',True,'res_word1_2.csv',False,'res')
train('chars',False,'res_char0_2.csv',False,'res')


"""
train('words',False,'res_word0_3.csv',False,'res')
train('chars',True,'res_char1_3.csv',False,'res')

train('words',True,'res_word1_3.csv',False,'res')
train('chars',False,'res_char0_3.csv',False,'res')
"""
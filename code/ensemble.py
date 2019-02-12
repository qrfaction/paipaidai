import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
import keras.backend as K
import multiprocessing as mlp

def ensemble(model_name,te_word,te_char,embedding_matrix_word,emembedding_matrix_char):
    from nn import rnnword, aggmodel, esim, attention, rnn_res

    if model_name == 'rnnword':
        get_model = rnnword
    elif model_name == 'aggmodel':
        pass
    elif model_name == 'esim':
        get_model = esim
    elif model_name == 'attention':
        get_model = attention
    elif model_name == 'res':
        get_model = rnn_res
    else:
        raise RuntimeError("don't have this model")

    path = './weight_' + model_name + '/'

    results = []
    m_char = get_model(emembedding_matrix_char,False)
    m_word = get_model(embedding_matrix_word,True)

    for model_path in tqdm(glob(path+'*.h5')):

        if "2018-07-15_16:15:17" not in model_path:
            continue
        if 'chars_True'in model_path or 'words_True' in model_path:
            ense_w = 7
        elif 'chars_False' in model_path:
            ense_w = 3
        elif 'words_False' in model_path:
            ense_w = 4
        else:
            raise RuntimeError("error model")

        if 'char' in model_path:
            m_char.load_weights(model_path)
            results.append((m_char.predict(te_char,batch_size=1024), ense_w))
        else:
            m_word.load_weights(model_path)
            results.append((m_word.predict(te_word,batch_size=1024),ense_w))

    K.clear_session()
    tf.reset_default_graph()

    submit = 0
    total_w = 0
    for y_pred,ense_w in results:
        submit += ense_w*y_pred
        total_w += ense_w

    return submit/total_w



if __name__ == '__main__':
    from readdata import read_data

    _, te_word, embedding_matrix_word,__ = read_data('words', data_aug=False)
    _, te_char, embedding_matrix_char,__ = read_data('chars', data_aug=False)

    submit_atten = ensemble('esim',te_word,te_char,embedding_matrix_word,embedding_matrix_char)

    submit = pd.DataFrame()
    submit['y_pre'] = list(submit_atten[:, 0])
    submit.to_csv('atten.csv', index=False)






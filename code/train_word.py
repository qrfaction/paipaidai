import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config import MAX_NB_WORDS,MAX_NUM_CHARS,MAX_NUM_WORDS
from tqdm import tqdm
import numpy as np
from keras.optimizers import Nadam


def get_embedd(word_index,file):
    embeddings_index = {}
    with open(file, 'r') as f:
        wordmat = f.read().split('\n')
        if wordmat[-1] == '':
            wordmat = wordmat[:-1]
        if wordmat[0] == '':
            wordmat = wordmat[1:]

    for line in tqdm(wordmat):
        wvec = line.strip('\n').strip(' ').split(' ')
        embeddings_index[wvec[0]] = np.asarray(wvec[1:], dtype='float')

    print('embedding', len(embeddings_index))

    EMBEDDING_DIM = 300
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(str(word).upper())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def get_model(word_embedding_matrix,char_embedding_matrix):
    from keras.models import Model
    from keras.layers import Input,Lambda,BatchNormalization
    from keras.layers import CuDNNGRU, Bidirectional,GlobalMaxPooling1D
    from keras.layers import Embedding,SpatialDropout1D,Dense
    from keras import backend as K


    def loss(y_true,y_pred):
        return -K.mean(y_pred * y_true)

    from config import MAX_NUM_WORDS,MAX_NUM_CHARS


    word = Input(shape=(MAX_NUM_WORDS,))
    char = Input(shape=(MAX_NUM_CHARS,))


    embedd_word = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=MAX_NUM_WORDS,
                   trainable=True,name='word_weight')
    embedd_char = Embedding(
        len(char_embedding_matrix),
        char_embedding_matrix.shape[1],
        weights=[char_embedding_matrix],
        input_length=MAX_NUM_CHARS,
        trainable=True,name='char_weight')

    gru_dim1 = 384

    gru_w = Bidirectional(CuDNNGRU(gru_dim1,return_sequences=True),merge_mode='sum')

    gru_c = Bidirectional(CuDNNGRU(gru_dim1, return_sequences=True), merge_mode='sum')

    w = embedd_word(word)
    c = embedd_char(char)
    w = BatchNormalization()(w)
    c = BatchNormalization()(c)
    w = SpatialDropout1D(0.2)(w)
    c = SpatialDropout1D(0.2)(c)

    w = gru_w(w)
    c = gru_c(c)

    w = GlobalMaxPooling1D()(w)
    c = GlobalMaxPooling1D()(c)

    def jaccard(x):
        x0_2 = K.sum(x[0] ** 2, axis=1, keepdims=True)
        x1_2 = K.sum(x[1] ** 2, axis=1, keepdims=True)
        x01_ = K.sum(K.abs(x[0] * x[1]), axis=1, keepdims=True)

        return x[0] * x[1]/(x0_2+x1_2-x01_)


    output = Lambda(jaccard)([w,c])
    output = Dense(1,activation='sigmoid')(output)
    model = Model(inputs=[word,char], outputs=output)

    model.compile(loss='binary_crossentropy',optimizer=Nadam())

    return model

def train():
    question = pd.read_csv('./data/question.csv')


    toke_word = Tokenizer(num_words=MAX_NB_WORDS)
    toke_word.fit_on_texts(question['words'])
    q_word = toke_word.texts_to_sequences(question['words'])
    q_word = pad_sequences(q_word, maxlen=MAX_NUM_WORDS, truncating='post')
    q_word = np.array(list(q_word)*2)
    word_index = toke_word.word_index
    word_embedd = get_embedd(word_index,'./data/word_embed.txt')


    toke_char = Tokenizer(num_words=MAX_NB_WORDS)
    toke_char.fit_on_texts(question['chars'])
    q_char = toke_word.texts_to_sequences(question['chars'])
    q_char = pad_sequences(q_char, maxlen=MAX_NUM_CHARS, truncating='post')
    q_char = np.array(list(q_char) + list(q_char)[::-1])
    char_index = toke_char.word_index
    char_embedd = get_embedd(char_index, './data/char_embed.txt')


    model = get_model(word_embedd,char_embedd)
    y = np.ones(len(q_char))
    y[len(question):] = 0
    model.fit([q_word,q_char],y,verbose=1,epochs=2,batch_size=512,shuffle=True)

    word_embedd = model.get_layer('word_weight').get_weights()
    char_embedd = model.get_layer('char_weight').get_weights()


    print('save  ')
    word_mat = ''
    for i in range(len(word_embedd)):
        w = word_index.get(i)
        if w is None:
            continue
        vec_str = ' '.join([w]+list(word_embedd[i]))
        vec_str+='\n'
        word_mat+=vec_str
    with open('./data/word_embed1.txt','w') as f:
        f.write(word_mat)



    char_mat = ''
    for i in range(len(char_embedd)):
        c = char_index.get(i)
        if c is None:
            continue
        vec_str = ' '.join([c] + list(char_embedd[i]))
        vec_str += '\n'
        char_mat += vec_str

    with open('./data/char_embed1.txt', 'w') as f:
        f.write(char_mat)


if __name__ == '__main__':
    train()






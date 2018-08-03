from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam,RMSprop
import tensorflow as tf
from keras.initializers import VarianceScaling
from itertools import combinations
from keras.constraints import non_neg,min_max_norm


def co_attention(q1, q2):

    dense_w = TimeDistributed(Dense(1))
    atten = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q1, Permute((2, 1))(q2)])   # 15 * 15

    atten_1 = dense_w(atten)
    atten_1 = Flatten()(atten_1)
    atten_1 = Activation('softmax')(atten_1)
    atten_1 = Reshape((1,-1))(atten_1)

    atten_2 = dense_w(Permute((2, 1))(atten))
    atten_2 = Flatten()(atten_2)
    atten_2 = Activation('softmax')(atten_2)
    atten_2 = Reshape((1,-1))(atten_2)

    q1 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_1,q1])   # 1*300
    q1 = Flatten()(q1)
    q2 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_2,q2])  # 1*300
    q2 = Flatten()(q2)
    return q1, q2

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape

def soft_attention_alignment(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: K.softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: K.softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def norm_layer(x, axis=1):
    return (x - K.mean(x, axis=axis, keepdims=True)) / K.std(x, axis=axis, keepdims=True)

def distance(q1,q2,dist,normlize=False):
    if normlize:
        q1 = Lambda(norm_layer)(q1)
        q2 = Lambda(norm_layer)(q2)

    if dist == 'cos':
        return multiply([q1,q2])

    elif dist == 'h_mean':
        def dice(x):
            return x[0]*x[1]/(K.sum(K.abs(x[0]),axis=1,keepdims=True)+K.sum(K.abs(x[1]),axis=1,keepdims=True))
        return Lambda(dice)([q1,q2])

    elif dist == 'dice':
        def dice(x):
            return x[0]*x[1]/(K.sum(x[0]**2,axis=1,keepdims=True)+K.sum(x[1]**2,axis=1,keepdims=True))
        return Lambda(dice)([q1,q2])

    elif dist == 'jaccard':
        def jaccard(x):
            return  x[0]*x[1]/(
                    K.sum(x[0]**2,axis=1,keepdims=True)+
                    K.sum(x[1]**2,axis=1,keepdims=True)-
                    K.sum(K.abs(x[0]*x[1]),axis=1,keepdims=True))
        return Lambda(jaccard)([q1,q2])
    elif dist == 'jac_add':
        def jac_add(x):
            a = K.sum(x[0]**2,axis=1,keepdims=True)+K.sum(x[1]**2,axis=1,keepdims=True)-K.sum(K.abs(x[0]*x[1]),axis=1,keepdims=True)
            b = x[0]+x[1]
            return  b/a
        return Lambda(jac_add)([q1,q2])
    elif dist == 'dice_add':
        def dice_add(x):
            a = K.sum(x[0]**2,axis=1,keepdims=True)+K.sum(x[1]**2,axis=1,keepdims=True)
            b = x[0]+x[1]
            return  b/a
        return Lambda(dice_add)([q1,q2])

def pool_corr(q1,q2,pool_way,dist):
    if pool_way == 'max':
        pool = GlobalMaxPooling1D()
    elif pool_way == 'ave':
        pool = GlobalAveragePooling1D()
    else:
        raise RuntimeError("don't have this pool way")

    q1 = pool(q1)
    q2 = pool(q2)

    merged = distance(q1,q2,dist,normlize=True)


    return merged

def weight_ave(q1,q2):

    down = TimeDistributed(Dense(1,use_bias=False))

    q1 = down(Permute((2,1))(q1))
    q1 = Flatten()(q1)
    q1 = Lambda(norm_layer)(q1)
    q2 = down(Permute((2,1))(q2))
    q2 = Flatten()(q2)
    q2 = Lambda(norm_layer)(q2)
    merged = multiply([q1, q2])
    return merged

def simility_vec(q1,q2):
    simi = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q1, Permute((2, 1))(q2)])
    simi = Reshape((-1,))(simi)
    return simi

def rnnword(word_embedding_matrix,use_word):
    if use_word:
        from config import MAX_NUM_WORDS
        text_len = MAX_NUM_WORDS
    else:
        from config import MAX_NUM_CHARS
        text_len = MAX_NUM_CHARS

    question1 = Input(shape=(text_len,),name='q1')
    question2 = Input(shape=(text_len,),name='q2')



    embedd_word = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=text_len,
                   trainable=True)


    gru_dim1 = 384
    gru_dim2 = 256


    gru_w = Bidirectional(CuDNNGRU(gru_dim1,return_sequences=True),merge_mode='sum')
    gru2_w = Bidirectional(CuDNNGRU(gru_dim2,return_sequences=True),merge_mode='sum')


    norm = BatchNormalization()
    q1 = embedd_word(question1)
    q1 = norm(q1)
    q1 = SpatialDropout1D(0.2)(q1)

    q2 = embedd_word(question2)
    q2 = norm(q2)
    q2 = SpatialDropout1D(0.2)(q2)

    q1_1 = gru_w(q1)
    q2_1 = gru_w(q2)

    q1 = gru2_w(q1_1)
    q2 = gru2_w(q2_1)

    merged_max = pool_corr(q1,q2,'max','jaccard')
    merged_ave = pool_corr(q1,q2,'ave','jaccard')

    from config import n_components
    q1_g = Input(shape=(n_components,),name='q1node')
    q2_g = Input(shape=(n_components,),name='q2node')


    norm = BatchNormalization()
    q1_node = norm(q1_g)
    q2_node = norm(q2_g)

    fc = Dense(units=2)
    act = PReLU()
    q1_node = fc(q1_node)
    q1_node = act(q1_node)
    q2_node = fc(q2_node)
    q2_node = act(q2_node)

    node_vec = multiply([q1_node,q2_node])

    graph_f = Input(shape=(11,),name='gf')
    gf = BatchNormalization()(graph_f)
    gf = Dropout(0.2)(gf)

    merged = concatenate([merged_ave,merged_max])
    merged = Dense(512,activation='relu')(merged)
    merged = concatenate([merged, gf,node_vec])
    merged = Dense(512,activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(merged)

    lr=0.0008

    model = Model(inputs=[question1,question2,graph_f,q1_g,q2_g], outputs=output)

    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['binary_crossentropy','accuracy'])
    print(lr)

    return model

def aggmodel(word_embedding_matrix,char_embedding_matrix):

    def prepocess(q1,q2,embedd):
        norm = BatchNormalization()
        q1 = embedd(q1)
        q1 = norm(q1)
        q1 = SpatialDropout1D(0.2)(q1)

        q2 = embedd(q2)
        q2 = norm(q2)
        q2 = SpatialDropout1D(0.2)(q2)
        return q1,q2

    from config import MAX_NUM_WORDS,MAX_NUM_CHARS


    word1 = Input(shape=(MAX_NUM_WORDS,))
    word2 = Input(shape=(MAX_NUM_WORDS,))
    char1 = Input(shape=(MAX_NUM_CHARS,))
    char2 = Input(shape=(MAX_NUM_CHARS,))


    embedd_word = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=MAX_NUM_WORDS,
                   trainable=True)
    embedd_char = Embedding(
        len(char_embedding_matrix),
        char_embedding_matrix.shape[1],
        weights=[char_embedding_matrix],
        input_length=MAX_NUM_CHARS,
        trainable=True)

    gru_dim1 = 384
    gru_dim2 = 256


    gru_w = Bidirectional(CuDNNGRU(gru_dim1,return_sequences=True),merge_mode='sum')
    gru2_w = Bidirectional(CuDNNGRU(gru_dim2,return_sequences=True,),merge_mode='sum')

    gru_wc = Bidirectional(CuDNNGRU(gru_dim1, return_sequences=True), merge_mode='sum')
    gru2_wc = Bidirectional(CuDNNGRU(gru_dim2, return_sequences=True), merge_mode='sum')

    q1,q2 = prepocess(word1,word2,embedd_word)
    qc1,qc2 = prepocess(char1,char2,embedd_char)

    q1 = gru_w(q1)
    q2 = gru_w(q2)
    qc1 = gru_wc(qc1)
    qc2 = gru_wc(qc2)

    q1 = gru2_w(q1)
    q2 = gru2_w(q2)
    qc1 = gru2_wc(qc1)
    qc2 = gru2_wc(qc2)

    merged_max1 = pool_corr(q1,qc2,'max')
    merged_max2 = pool_corr(qc1,q2,'max')
    merged_ave1 = pool_corr(q1,qc2,'ave')
    merged_ave2 = pool_corr(qc1,q2,'ave')

    merged_max3 = pool_corr(q1,q2, 'max')
    merged_max4 = pool_corr(qc1,qc2, 'max')
    merged_ave3 = pool_corr(q1,q2, 'ave')
    merged_ave4 = pool_corr(qc1,qc2, 'ave')


    merged = concatenate([merged_max1,merged_max2,merged_max3,merged_max4,
                          merged_ave1,merged_ave2,merged_ave3,merged_ave4])
    merged = Dense(512,activation='relu')(merged)
    # merged = Dropout(0.2)(merged)
    merged = Dense(512,activation='relu')(merged)
    # merged = Dropout(0.2)(merged)
    output = Dense(1, activation='sigmoid')(merged)



    lr=0.0008


    model = Model(inputs=[word1,word2,char1,char2], outputs=output)

    # model = multi_gpu_model(model,gpus=4)

    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['binary_crossentropy','accuracy'])

    # model.load_weights("./data/weights_best_0.0008.hdf5")
    print(lr)

    return model

def esim(word_embedding_matrix, use_word):
    if use_word:
        from config import MAX_NUM_WORDS
        text_len = MAX_NUM_WORDS
    else:
        from config import MAX_NUM_CHARS
        text_len = MAX_NUM_CHARS

    q1 = Input(name='q1', shape=(text_len,))
    q2 = Input(name='q2', shape=(text_len,))

    embedding = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=text_len,
                   trainable=True)

    bn = BatchNormalization()
    q1_embed = bn(embedding(q1))
    q1_embed = SpatialDropout1D(0.2)(q1_embed)
    q2_embed = bn(embedding(q2))
    q2_embed = SpatialDropout1D(0.2)(q2_embed)

    encode = Bidirectional(CuDNNLSTM(384,return_sequences=True), merge_mode='sum')
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    q1_combined = Concatenate()([q1_encoded, q2_aligned, multiply([q1_encoded, q2_aligned])])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, multiply([q2_encoded, q1_aligned])])

    compose = Bidirectional(CuDNNLSTM(384,return_sequences=True), merge_mode='sum')
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)


    merged_ave = pool_corr(q1_compare,q2_compare,'ave','dice')
    merged_max = pool_corr(q1_compare,q2_compare,'max','dice')

    from config import n_components
    q1_g = Input(shape=(n_components,), name='q1node')
    q2_g = Input(shape=(n_components,), name='q2node')

    norm = BatchNormalization()
    q1_node = norm(q1_g)
    q2_node = norm(q2_g)

    fc = Dense(units=2)
    act = PReLU()
    q1_node = fc(q1_node)
    q1_node = act(q1_node)
    q2_node = fc(q2_node)
    q2_node = act(q2_node)

    node_vec = multiply([q1_node, q2_node])

    graph_f = Input(shape=(11,), name='gf')
    gf = BatchNormalization()(graph_f)
    gf = Dropout(0.2)(gf)

    merged = Concatenate()([merged_max, merged_ave])

    dense = Dense(512, activation='relu')(merged)
    dense = concatenate([dense,gf,node_vec])
    dense = Dense(512, activation='relu')(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
    lr = 0.0008

    model = Model(inputs=[q1, q2, graph_f,q1_g,q2_g], outputs=out_)
    model.compile(optimizer=Nadam(lr=lr), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
    return model

def attention(word_embedding_matrix,use_word):
    if use_word:
        from config import MAX_NUM_WORDS
        text_len = MAX_NUM_WORDS
    else:
        from config import MAX_NUM_CHARS
        text_len = MAX_NUM_CHARS

    question1 = Input(shape=(text_len,),name='q1')
    question2 = Input(shape=(text_len,),name='q2')



    embedd_word = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=text_len,
                   trainable=True)

    gru_dim1 = 300
    gru_dim2 = 300

    gru_w = Bidirectional(CuDNNLSTM(gru_dim1,return_sequences=True),merge_mode='sum')
    gru2_w = Bidirectional(CuDNNLSTM(gru_dim2,return_sequences=True),merge_mode='sum')


    norm = BatchNormalization()
    q1 = embedd_word(question1)
    q1 = norm(q1)
    q1 = SpatialDropout1D(0.2)(q1)

    q2 = embedd_word(question2)
    q2 = norm(q2)
    q2 = SpatialDropout1D(0.2)(q2)

    q1 = gru_w(q1)
    q2 = gru_w(q2)

    q1 = gru2_w(q1)
    q2 = gru2_w(q2)

    q1_1,q2_2 = co_attention(q1,q2)
    merged_1 = distance(q1_1,q2_2,'dice', normlize=True)
    merged_3 = pool_corr(q1,q2,'max','dice')
    merged_4 = distance(q1_1,q2_2,'dice_add',normlize=True)

    from config import n_components
    q1_g = Input(shape=(n_components,),name='q1node')
    q2_g = Input(shape=(n_components,),name='q2node')

    norm = BatchNormalization()
    q1_node = norm(q1_g)
    q2_node = norm(q2_g)

    fc = Dense(units=2)
    act = PReLU()
    q1_node = fc(q1_node)
    q1_node = act(q1_node)
    q2_node = fc(q2_node)
    q2_node = act(q2_node)

    node_vec = multiply([q1_node,q2_node])

    graph_f = Input(shape=(11,),name='gf')
    gf = BatchNormalization()(graph_f)
    gf = Dropout(0.2)(gf)

    merged = concatenate([merged_1,merged_3,merged_4])
    merged = Dense(768,activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = concatenate([merged,gf,node_vec])
    merged = Dense(768,activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(merged)

    lr=0.0008

    model = Model(inputs=[question1,question2,graph_f,q1_g,q2_g], outputs=output)

    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['binary_crossentropy','accuracy'])
    print(lr)

    return model


def rnn_res(word_embedding_matrix,use_word):
    if use_word:
        from config import MAX_NUM_WORDS
        text_len = MAX_NUM_WORDS
    else:
        from config import MAX_NUM_CHARS
        text_len = MAX_NUM_CHARS

    question1 = Input(shape=(text_len,),name='q1')
    question2 = Input(shape=(text_len,),name='q2')



    embedd_word = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=text_len,
                   trainable=True)

    gru_dim1 = 300
    gru_dim2 = 300

    gru_w = Bidirectional(CuDNNLSTM(gru_dim1,return_sequences=True),merge_mode='sum')
    gru2_w = Bidirectional(CuDNNGRU(gru_dim2,return_sequences=True),merge_mode='sum')


    norm = BatchNormalization()
    q1 = embedd_word(question1)
    q1 = norm(q1)
    q1 = SpatialDropout1D(0.2)(q1)

    q2 = embedd_word(question2)
    q2 = norm(q2)
    q2 = SpatialDropout1D(0.2)(q2)

    q1_0 = gru_w(q1)
    q2_0 = gru_w(q2)

    q1 = gru2_w(q1_0)
    q2 = gru2_w(q2_0)

    merged_0 = pool_corr(q1_0,q2_0,'ave','jaccard')
    merged_1 = pool_corr(q1,q2,'ave','dice')
    merged_2 = pool_corr(q1_0, q2_0, 'max', 'jaccard')
    merged_3 = pool_corr(q1,q2,'max','dice')

    from config import n_components
    q1_g = Input(shape=(n_components,),name='q1node')
    q2_g = Input(shape=(n_components,),name='q2node')

    norm = BatchNormalization()
    q1_node = norm(q1_g)
    q2_node = norm(q2_g)

    fc = Dense(units=2)
    act = PReLU()
    q1_node = fc(q1_node)
    q1_node = act(q1_node)
    q2_node = fc(q2_node)
    q2_node = act(q2_node)

    node_vec = multiply([q1_node,q2_node])

    graph_f = Input(shape=(11,),name='gf')
    gf = BatchNormalization()(graph_f)
    gf = Dropout(0.2)(gf)

    merged = concatenate([merged_1,merged_3,merged_2,merged_0])
    merged = Dense(768,activation='relu')(merged)
    merged = concatenate([merged, gf,node_vec])
    merged = Dense(768,activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(merged)

    lr=0.0008

    model = Model(inputs=[question1,question2,graph_f,q1_g,q2_g], outputs=output)

    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['binary_crossentropy','accuracy'])
    print(lr)

    return model

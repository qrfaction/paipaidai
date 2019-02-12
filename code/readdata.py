
import pandas as pd
import numpy as np
# 文本处理
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config import MAX_NB_WORDS
from tqdm import tqdm
from tool import get_samples
from sklearn.preprocessing import LabelEncoder
# 20890

def get_embedding_matrix(word_index,file):
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
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(str(word).upper())
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def read_data(use_data,file=None,data_aug=False):

    question = pd.read_csv('./data/question.csv')
    question = question[['qid', use_data]]


    if data_aug:
        train = pd.read_csv('./data/train.csv', usecols=['label', 'q1', 'q2'])
        samples = pd.read_csv('./data/aug_data_filter.csv',usecols=['label','q1','q2'])
        train = pd.concat([train,samples]).reset_index(drop=True)
        test = pd.read_csv('./data/test.csv', usecols=['q1', 'q2'])
    else:
        train = pd.read_csv('./data/train.csv', usecols=['label', 'q1', 'q2'])
        test = pd.read_csv('./data/test.csv', usecols=['q1', 'q2'])


    train = pd.merge(train, question, left_on=['q1'], right_on=['qid'], how='left')
    train = pd.merge(train, question, left_on=['q2'], right_on=['qid'], how='left')
    train = train[[use_data+'_x', use_data+'_y','label']]
    train.columns = ['q1', 'q2','label']

    test = pd.merge(test, question, left_on=['q1'], right_on=['qid'], how='left')
    test = pd.merge(test, question, left_on=['q2'], right_on=['qid'], how='left')
    test = test[[use_data+'_x', use_data+'_y']]
    test.columns = ['q1', 'q2']

    all = pd.concat([train, test])

    # 分词 词转序列
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(question[use_data])

    word_index = tokenizer.word_index
    print(len(word_index))

    q1_word_seq = tokenizer.texts_to_sequences(all['q1'])
    q2_word_seq = tokenizer.texts_to_sequences(all['q2'])

    if file is None:
        if use_data == 'words':
            file = './data/word_embed.txt'
        if use_data == 'chars':
            file = './data/char_embed.txt'
    word_embedding_matrix = get_embedding_matrix(word_index, file)


    from config import MAX_NUM_WORDS,MAX_NUM_CHARS
    if use_data == 'words':
        text_len = MAX_NUM_WORDS
    elif use_data == 'chars':
        text_len = MAX_NUM_CHARS
    else:
        raise RuntimeError('use data error')

    q1_data = pad_sequences(q1_word_seq,maxlen=text_len,truncating='post')
    q2_data = pad_sequences(q2_word_seq,maxlen=text_len,truncating='post')

    tr_q1 = q1_data[:train.shape[0]]
    tr_q2 = q2_data[:train.shape[0]]

    te_q1 = q1_data[train.shape[0]:]
    te_q2 = q2_data[train.shape[0]:]

    usecols = [
        'q1q2_union_w',
        'q1q2_inter_w',
        'q1_num_adj_w',
        'q2_num_adj_w',
        'q1q2_union',
        'q1q2_inter',
        'q1_num_adj',
        'q2_num_adj',
        # 'q1_hash',
        # 'q2_hash',
        # 'q1q2_inter',
        'shortest_path_w',
        # 'edit_words',
        # 'edit_chars',
        # 'lcs_words',
        # 'lcs_chars',
        # 'q1_word_len',
        # 'q2_word_len',
        # 'q1_char_len',
        # 'q2_char_len',
        # 'words_common',
        # 'chars_common',
    ]
    # if data_aug==False:
    usecols+=['q1_pr_w','q2_pr_w']

    tr = {}
    tr['q1'] = tr_q1
    tr['q2'] = tr_q2
    te = {}
    te['q1'] = te_q1
    te['q2'] = te_q2

    if data_aug:
        tr['gf'] = pd.concat([pd.read_csv('./data/train.csv',usecols=usecols),
                              pd.DataFrame(np.zeros((len(samples),11)),columns=usecols)]).values
        te['gf'] = pd.read_csv('./data/test.csv', usecols=usecols).values
    else:
        tr['gf'] = pd.read_csv('./data/train.csv', usecols=usecols).values
        te['gf'] = pd.read_csv('./data/test.csv', usecols=usecols).values


    if data_aug:
        q_tr = pd.concat([pd.read_csv('./data/train.csv',usecols=['q1','q2']),
                          pd.read_csv('./data/aug_data_filter.csv',usecols=['q1','q2'])]).reset_index(drop=True)
        q_te = pd.read_csv('./data/test.csv', usecols=['q1', 'q2'])
    else:
        q_tr = pd.read_csv('./data/train.csv',usecols=['q1','q2'])
        q_te = pd.read_csv('./data/test.csv',usecols=['q1','q2'])

    if data_aug:
        questions = pd.read_csv('./data/q_matrix.csv')
    else:
        questions = pd.read_csv('./data/q_matrix.csv')

    from config import n_components
    feat = ["feat"+str(i) for i in range(n_components)]

    tr['q1node'] = pd.merge(q_tr, questions, left_on=['q1'], right_on=['qid'], how='left').loc[:,feat].values
    tr['q2node'] = pd.merge(q_tr, questions, left_on=['q2'], right_on=['qid'], how='left').loc[:,feat].values
    te['q1node'] = pd.merge(q_te, questions, left_on=['q1'], right_on=['qid'], how='left').loc[:,feat].values
    te['q2node'] = pd.merge(q_te, questions, left_on=['q2'], right_on=['qid'], how='left').loc[:,feat].values

    # q_embed = questions.loc[:,['feat'+str(i) for i in range(128)]].values

    return tr,te, word_embedding_matrix,train['label']

def save_data_tree(use_data,file=None):
    question = pd.read_csv('./data/question.csv')
    question = question[['qid', use_data]]

    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    train = pd.merge(train, question, left_on=['q1'], right_on=['qid'], how='left')
    train = pd.merge(train, question, left_on=['q2'], right_on=['qid'], how='left')
    train = train[[use_data + '_x', use_data + '_y', 'label']]
    train.columns = ['q1', 'q2', 'label']

    test = pd.merge(test, question, left_on=['q1'], right_on=['qid'], how='left')
    test = pd.merge(test, question, left_on=['q2'], right_on=['qid'], how='left')
    test = test[[use_data + '_x', use_data + '_y']]
    test.columns = ['q1', 'q2']

    all = pd.concat([train, test])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(question[use_data])

    word_index = tokenizer.word_index
    print(len(word_index))

    q1_word_seq = tokenizer.texts_to_sequences(all['q1'])
    q2_word_seq = tokenizer.texts_to_sequences(all['q2'])

    if file is None:
        if use_data == 'words':
            file = './data/word_embed.txt'
        if use_data == 'chars':
            file = './data/char_embed.txt'
    embedding_matrix = get_embedding_matrix(word_index, file)

    from config import MAX_NUM_WORDS, MAX_NUM_CHARS
    if use_data == 'words':
        text_len = MAX_NUM_WORDS
    elif use_data == 'chars':
        text_len = MAX_NUM_CHARS
    else:
        raise RuntimeError('use data error')

    q1_data = pad_sequences(q1_word_seq, maxlen=text_len, truncating='post')
    q2_data = pad_sequences(q2_word_seq, maxlen=text_len, truncating='post')

    q1_matrix = np.zeros((len(q1_data),300*text_len),dtype=np.float16)
    q2_matrix = np.zeros((len(q2_data),300*text_len),dtype=np.float16)

    embedding_matrix = embedding_matrix.astype(np.float16)
    for i,(q1,q2) in tqdm(enumerate(zip(q1_data,q2_data))):
        for j in range(text_len):
            if q1[j] != 0:
                w_v = embedding_matrix[q1[j]]
                q1_matrix[i,j*300:(j+1)*300] = w_v
            if q2[j] != 0:
                w_v = embedding_matrix[q2[j]]
                q2_matrix[i,j*300:(j+1)*300] = w_v

    from scipy.sparse import csr_matrix,save_npz

    save_npz('./data/q1_matrix'+use_data+'.npz',csr_matrix(q1_matrix))
    save_npz('./data/q2_matrix'+use_data+'.npz',csr_matrix(q2_matrix))

    print('success')

if __name__ == '__main__':
    save_data_tree('words')
    save_data_tree('chars')







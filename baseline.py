import pandas as pd
import nltk

data = pd.read_csv('C:/Users/Prudhvi/repos/factoid/data/question_answer_pairs.txt' ,sep ='\t' , encoding= 'unicode_escape')

data_new = data.drop_duplicates(subset = 'Question')
data_new.to_csv('data.csv')

all_qforms = ['where','what','when','who','how','which','whom','why','is','was','did','are','does','do','can','whom']


def find_qform(row):
    tokens = [word.lower() for word in str(row['Question']).split()]
    qform = list(set(tokens) & set(all_qforms))
    if qform:
        return qform[0]
    else:
        return 'Not found'



data_new['qform'] = data_new.apply(lambda x : find_qform(x), axis = 1)

data_new['qform'].value_counts()
data_new['qform'].value_counts().sum()

data_new['Answer'] = data_new['Answer'].astype('category')


def yes_no(row):
    l = ['yes','no']
    tokens = [word.lower() for word in nltk.word_tokenize(str(row['Answer']))]
    common = list(set(tokens) & set(l))
    if common:
        return common[0]
    else:
        return 'Not found'

data_new['ans_yn'] = data_new.apply(lambda x : yes_no(x), axis = 1)
data_new['ans_yn'].value_counts()

yn_data = data_new[data_new['ans_yn'].isin(['yes','no'])]
data_new.reset_index(drop = True, inplace= True)

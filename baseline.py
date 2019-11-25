import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

# Reading Datasets
data = pd.read_csv(r'./data/question_answer_pairs.txt' ,sep ='\t' , encoding= 'unicode_escape')
data2 = pd.read_csv(r'./data/question_answer_pairs2.txt' ,sep ='\t' , encoding= 'unicode_escape')
data3 = pd.read_csv(r'./data/question_answer_pairs3.txt' ,sep ='\t' , encoding= 'unicode_escape')

data2.ArticleFile= data2.ArticleFile.apply(lambda x: x.replace("data", "data2", 1))
data3.ArticleFile= data3.ArticleFile.apply(lambda x: x.replace("data", "data3", 1))

# Appending all the dataset into a single dataset
data= data.append(data2, ignore_index= True)
data= data.append(data3, ignore_index= True)

# Droping Duplicates
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

# Reseting the index
yn_data.reset_index(inplace= True, drop= True)
# Imputing NULL
yn_data.ArticleFile[111]= 'data/set3/a10'

# Reading Articles into dataframe
yn_data['Article']= ""
for i in range(yn_data.shape[0]):
    with open(f'data/{yn_data.ArticleFile[i]}.txt', 'r', encoding= 'unicode_escape') as f:
        yn_data['Article'][i]= f.read()

#Cleaning Text
for i in range(8):
    yn_data.Article[i]= yn_data.Article[i][:60760]

#Sentence Segmentation of Article Text
yn_data.Article= yn_data.Article.apply(sent_tokenize)

# Word Share or Unigram Match
def word_share(sen1, sen2):
    sen1= word_tokenize(sen1.lower())
    sen2= word_tokenize(sen2.lower())
    comm= list(set(sen1) & set(sen2))
    return len(comm)/len(sen1)

# Getting Matched Sentence using UniGram Match
yn_data['matched_sentence']= ""
for i in range(yn_data.shape[0]):
    word_shares= list()
    for sentence in yn_data.Article[i]:
        word_shares.append(word_share(yn_data.Question[i], sentence))
    yn_data['matched_sentence'][i]= yn_data.Article[i][word_shares.index(max(word_shares))]

# copying the relevent data in a diff dataframe
df= yn_data[['Question', 'matched_sentence', 'qform', 'ans_yn']].copy()
df.to_csv('final_data.csv', index= False)

# Tokenizing Text
df.Question= df.Question.apply(lambda x: word_tokenize(x.lower()))
df.matched_sentence= df.matched_sentence.apply(lambda x: word_tokenize(x.lower()))

############ Getting Adjective Match ############
def get_adj(text):
    tags= nltk.pos_tag(text, tagset= 'universal')
    adj_tags= [t for t in tags if t[1] == "ADJ"]
    return [tags[0] for tags in adj_tags]

df['ques_adj']= df.Question.apply(get_adj)
df['match_adj']= df.matched_sentence.apply(get_adj)

df['adjective_match']= ""
for i in range(df.shape[0]):
    if any(item in df.match_adj[i] for item in df.ques_adj[i]):
        df['adjective_match'][i]= 1
    else:
        df['adjective_match'][i]= 0
###################################################

############ Getting Noun Match ###################
def get_noun(text):
    tags= nltk.pos_tag(text, tagset= 'universal')
    noun_tags= [t for t in tags if t[1] == "NOUN"]
    return [tags[0] for tags in noun_tags]

df['ques_noun']= df.Question.apply(get_noun)
df['match_noun']= df.matched_sentence.apply(get_noun)
df['noun_match']= ""
for i in range(df.shape[0]):
    if any(item in df.match_noun[i] for item in df.ques_noun[i]):
        df['noun_match'][i]= 1
    else:
        df['noun_match'][i]= 0
#####################################################

############# Getting Adverb Match #################
def get_adv(text):
    tags= nltk.pos_tag(text, tagset= 'universal')
    adv_tags= [t for t in tags if t[1] == "ADV"]
    return [tags[0] for tags in adv_tags]
df['ques_adv']= df.Question.apply(get_adv)
df['match_adv']= df.matched_sentence.apply(get_adv)
df['adv_match']= ""
for i in range(df.shape[0]):
    if any(item in df.match_adv[i] for item in df.ques_adv[i]):
        df['adv_match'][i]= 1
    else:
        df['adv_match'][i]= 0
#####################################################

############# Getting verb Match #####################
def get_verb(text):
    tags= nltk.pos_tag(text, tagset= 'universal')
    verb_tags= [t for t in tags if t[1] == "VERB"]
    return [tags[0] for tags in verb_tags]
df['ques_verb']= df.Question.apply(get_adv)
df['match_verb']= df.matched_sentence.apply(get_adv)
df['verb_match']= ""
for i in range(df.shape[0]):
    if all(item in df.match_verb[i] for item in df.ques_verb[i]):
        df['verb_match'][i]= 1
    else:
        df['verb_match'][i]= 0
######################################################

################## Extracting NERs ##################
import spacy
# python -m spacy download en_core_web_lg
nlp= spacy.load('en_core_web_lg')
ners_spacy= ['PERSON' ,'NORP' ,'FAC' ,'ORG' ,'GPE' ,'LOC' ,
             'PRODUCT' ,'EVENT' ,'WORK_OF_ART' ,'LAW' ,'LANGUAGE' ,'DATE' ,'TIME' ,
             'PERCENT' ,'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
for ner in ners_spacy:
    df[f'{ner}_ner_match']= ''
    for i in range(df.shape[0]):
        doc1= nlp(df.Question[i])
        doc2= nlp(df.matched_sentence[i])
        doc1_NER= list()
        for ent in doc1.ents:
            if ent.label_ == ner:
                doc1_NER.append(ent)
        doc2_NER= list()
        for ent in doc2.ents:
            if ent.label_ == ner:
                doc2_NER.append(ent)
        if any(item in doc2_NER for item in doc1_NER):
            df[f'{ner}_ner_match'][i]= 1
        else:
            df[f'{ner}_ner_match'][i]= 0
#######################################################

# Independent and Dependent Variable
X = df[['adjective_match', 'noun_match','adv_match','verb_match']]
y = df.ans_yn.map({'yes': 1, 'no': 0})


# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify= y, random_state= 42)

# Fitting Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred)) # tp / (tp + fp)
print('recall:', recall_score(y_test, y_pred)) # tp / (tp + fn)
print('f1-score:', f1_score(y_test, y_pred))

# Printing the Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')


# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

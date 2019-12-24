import pandas as pd
import numpy as np
import spacy
from scipy.spatial import distance
import edit_distance
import random

nlp = spacy.load('en_core_web_lg')

"""
job: return list of words with nlp.similar

input: str, int
output: list
"""
def get_similar_words(input_word,num_of_words):
  #print('get_similar_words')
  global vectors,ids

  p = np.array([nlp.vocab[input_word].vector])
  closest_index = distance.cdist(p, vectors)
  #print('closest_index',closest_index)
  output_list=[]
  closest_indexes = closest_index.argsort()
  #print('closest_indexes',closest_indexes)
  closest_indexes = np.squeeze(closest_indexes)
  closest_indexes = closest_indexes[0:105]
  for i in closest_indexes:
    word_id = ids[i]
    output_word = nlp.vocab[word_id]
    output_word = output_word.text.lower()
    #print('in',type(input_word))
    #print('out',type(output_word))
    sm = edit_distance.SequenceMatcher(input_word.lower(), output_word.lower())
    levin_dist = sm.distance()
    if ( (output_word.lower() != input_word.lower() ) and (levin_dist >2) ) :
      output_word = output_word
      output_list.append(output_word)
      if len(output_list) >= num_of_words:
        return output_list
  return output_list

"""
job: replace words with nlp.similar word with same POS

input: list  - [token.text,token.pos_,token.i]
output: bool,str,int
"""
def replace_noun(original_noun):
  # print('replace_noun')
  global vectors,ids
  #print(original_noun)
  original_word,original_pos,original_i = original_noun
  # print(type(original_word))
  replaced = False
  i=0
  j= -1
  while (replaced == False) and (i<50):
    i+=1
    #print('i',i)
    word_options = get_similar_words(original_word,10)
    # print('word_options',word_options)
    #word_options = w2v.wv.similar_by_word(original_word , topn=5*i)
    same_pos=False
    same_word=True
    while ((same_word==True) and (same_pos == False) and (j<(len(word_options)-1)) and (j<50)):
      j +=1
      new_word =  word_options[j]
      # print('word_options[j]',new_word)
      new_word = nlp(new_word)
      # print('new_word[0]',new_word)
      if new_word[0].text != original_word: same_word=False
      pos = new_word[0].pos_
      if pos == original_pos: same_pos = True
    replaced = True
  return replaced, new_word, original_i

"""
job: replace words by noun_chunks with nlp.similar words with same POS

input: str,float(0,1)
output: str
"""
def replace_noun_chunks_text(text, percent):
    #print('replace_noun_chunks_text')
    k = -1
    text = nlp(text)
    noun_ls = [[token.root.text, token.root.pos_, token.root.i] for token in text.noun_chunks]
    noun_size = len(noun_ls)
    noun_num = int(noun_size * percent)
    indexes = np.random.randint(0, noun_size, noun_num)
    original_nouns = [noun_ls[i] for i in indexes]
    original_indexes = [noun_ls[i][2] for i in indexes]
    # [original_noun[2] for original_noun in original_nouns ]

    new_text = []
    for i, token in enumerate(text[0:-2]):

        # i=int(i)
        # print(i)
        # print(i,i in original_indexes)
        if i in original_indexes:
            k += 1
            # print('yes')
            original_noun = original_nouns[k]
            replaced, new_word, original_i = replace_noun(original_noun)
            if replaced:
                # print(original_noun,new_word)
                new_text.append(new_word.text)
            else:
                # print('not')
                new_text.append(original_noun)

        else:
            # print('not')
            new_text.append(token.text)

    new_text = " ".join(new_text)
    # new_nouns = [replace_noun(original_noun) for original_noun in  original_nouns]

    return new_text

"""
job: replace words with nlp.similar words of same POS

input: str,str,float(0,1)
output: str
"""
def replace_pos_text(text, pos_str, percent):
    #print('replace_pos_text',pos_str)
    original_words = []
    original_indexes = []
    k = -1
    text = nlp(text)
    # TODO
    words_ls = [[token.text, token.pos_, token.i] for token in text if token.pos_ == pos_str]
    adjective_size = len(words_ls)
    adjective_num = int(adjective_size * percent)
    indexes = np.random.randint(0, adjective_size, adjective_num)
    for i in indexes:
        original_words.append(words_ls[i])
        original_indexes.append(words_ls[i][2])
    # original_words = [words_ls[i] for i in indexes]

    new_text = []
    for i, token in enumerate(text):

        # i=int(i)
        # print(i)
        # print(i,i in original_indexes)
        if i in original_indexes:
            k += 1
            # print('yes')
            original_word = original_words[k]
            replaced, new_word, original_i = replace_noun(original_word)
            if replaced:
                # print(original_noun,new_word)
                new_text.append(new_word.text)
            else:
                # print('not')
                new_text.append(original_word)

        else:
            # print('not')
            new_text.append(token.text)

    new_text = " ".join(new_text)
    # new_nouns = [replace_noun(original_noun) for original_noun in  original_nouns]

    return new_text


def create_vectors():
  ids = [x for x in nlp.vocab.vectors.keys()]
  vectors = [nlp.vocab.vectors[x] for x in ids]
  vectors = np.array(vectors)
  return vectors,ids


def my_split(data,precent):
  index = int(len(data)*precent)
  data = data.sample(frac=1)
  train_df = data.iloc[0:index,:]
  test_df = data.iloc[index::,:]
  return train_df,test_df


def expand_df(data, by_how_much):
    new_data = pd.DataFrame()
    for row_index in range(len(data)):
        if row_index % 1 == 0: print(row_index, 'of ', len(data))
        for i in range(by_how_much):
            new_row = data.iloc[row_index, :].copy()
            text = new_row['text']
            percent = random.random()

            # augmentation opptions:
            # rand joind text
            # new_text = get_random_joined_text(text,percent)

            # skip rand words
            # new_text = skip_rand_words(text,percent)

            # add_rand_words
            # text_index = random.randint(0,len(data))
            # print(text_index)
            # rand_text = data.iloc[text_index,:]['text']
            # print(rand_text)
            # new_text = add_rand_words(rand_text,text,percent)

            # replace by noun chunks
            # new_text = replace_noun_chunks_text(text, percent)

            # replace by pos_
            new_text = replace_pos_text(text, 'ADJ', percent)
            percent = random.random()
            new_text = replace_pos_text(new_text, 'PRON', percent)

            new_row['text'] = new_text
            new_data = new_data.append(new_row.copy())

    return new_data


if __name__ == '__main__':
    vectors, ids = create_vectors()  # nlp vocab
    # input
    df = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\mini_train_hotels.csv')

    if 'Unnamed: 0' in df.columns.values:
        df.drop('Unnamed: 0',axis=1,inplace=True)
    print(df.shape)


    new_data = expand_df(df, by_how_much=2)
    new_data.drop_duplicates(subset='text',inplace=True)
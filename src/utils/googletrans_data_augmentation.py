""" googletrans_data_augmentation.py create translations through googletrans library.
    Because of googletrans limitations on daily usage and maximum length per translation
    DataFrame is split according to daily charlimit of specified column.
    After user specifies chunk for translation. The part is divided yet again to smaller chunks for translation.
    translations are added to specified column of original dataframe and saved to new csv file.
    Best way to use :
    first run : chunk_to_translate= 0 text_column='text', translation_column='fr_trans', src_lang='en', dest_lang='fr')
    change out-put file name
    second run : chunk_to_translate= 0 text_column='fr_trans', translation_column='frde_trans', src_lang='fr', dest_lang='de')
    change out-put file name
    third run : chunk_to_translate=0 text_column='frde_trans', translation_column='frde_enbt', src_lang='de', dest_lang='en')
    change out-put file name
    change out-put file name and repeat the same 3 steps with chunk_to_translate= 1 and so on ...

    If translation fails during run:
     1. write part and chunk number.
     2. Insert chunk number to resume (instead of default resume =  False)
     3. Run program again - on same part after 25 hours or after reset to IP.
        GoodLuck """

import pandas as pd
from googletrans import Translator
import time
import sys
import numpy as np


def translate(origin, src, dest):
    time.sleep(np.random.randint(5, 15))
    # check if origin is string or list. - if string return [string]
    origin_list = [origin] if isinstance(origin, str) else origin
    translator = Translator()
    translates_obj = translator.translate(origin_list, src=src, dest=dest)
    # create a list texts from the objects returned by translator
    translates_list = list(map(lambda x: x.text, translates_obj))
    return translates_list


def translate_test(origin, src, dest):  # for testing without real translation
    origin_list = [origin] if isinstance(origin, str) else origin
    translates_obj = origin_list
    translates_list = list(map(lambda x: "1" + x, translates_obj))
    return translates_list

def chunk_texts_in_dataset(dataset, text_coulmn, char_limit):  # create llists with chunks of data from df columns
    all_char_counter = 0  # counter for al characters in text
    char_counter = 0  # counter for characters in chunk
    doc_ids_chunk = []  # applying until chunk is full
    texts_chunk = []  # applying until chunk is full
    list_id_df = []  # A list to keep chunks of doc_ids
    list_text_df = []  # A list to keep chunks of texts
    last_text_index = []  # A list to keep index of last instance in chunk
    # create a list of doc_ids from column
    doc_ids = dataset['document_id'].tolist()
    # create a list of texts from column
    texts = dataset[text_coulmn].tolist()
    # check char length of longest text in list
    max_text_len = len(max(texts, key=len))
    print('longest text has ', max_text_len, ' characters.')
    # check if there are texts longer then char limit - if so prompt
    if max_text_len >= char_limit:
        print(
            "longest text has {} chars.\n"
            " Texts longer then char_limit of {} characters, will loose the part after char_limit".format(
                max_text_len, char_limit))
    # start chunking
    # iterate over texts
    for texts_index in range(len(texts)):
        # if chunk has room for next text :count chars, add doc_id , add text
        if (char_counter + len(texts[texts_index])) <= char_limit:
            char_counter += len(texts[texts_index])
            doc_ids_chunk.append(doc_ids[texts_index])
            texts_chunk.append(texts[texts_index])
            # print (texts_index)
        # if text is longer then char_limit -close previous chunk, cut text and put it's first part in a new chunk
        elif len(texts[texts_index]) >= char_limit:
            print('''document_id: {} at index {} is {} chars long and exceeds char limit.\n
             it will lose the last part of it's data'''.format(doc_ids[texts_index], texts_index,
                                                               len(texts[texts_index])))
            list_id_df.append(doc_ids_chunk)
            list_text_df.append(texts_chunk)
            last_text_index.append(texts_index)
            all_char_counter += char_counter
            char_counter = 0
            texts_chunk = []
            doc_ids_chunk = []
            char_counter += len(texts[texts_index])
            doc_ids_chunk.append(doc_ids[texts_index])
            texts_chunk.append(texts[texts_index][:char_limit])
            # print (texts_index)
        else:  # rached char_limit - apply df lists and restart chunk lists and counter
            list_id_df.append(doc_ids_chunk)
            list_text_df.append(texts_chunk)
            last_text_index.append(texts_index)
            all_char_counter += char_counter
            print("chunk {} |  last index in chunk is {}".format(len(list_id_df) - 1, texts_index - 1))
            char_counter = 0
            texts_chunk = []
            doc_ids_chunk = []
            char_counter += len(texts[texts_index])
            doc_ids_chunk.append(doc_ids[texts_index])
            texts_chunk.append(texts[texts_index])
    print("* Chunk {} |   last index in chunk is {}. Last chunk for this dataset".format(len(list_id_df), texts_index))
    # finished iteration over all instances. apply final chunk to df lists.
    last_text_index.append(
        texts_index - 1)  # apply -1 so first instance in last chunk is not a duplicate of last instance in previous chunk
    all_char_counter += char_counter
    list_id_df.append(doc_ids_chunk)
    list_text_df.append(texts_chunk)
    print("dataset is {} characters long.".format(all_char_counter))
    return last_text_index, list_id_df, list_text_df


def create_translation_df(dataset, text_column, translation_column, src_lang, dest_lang, char_limit, output_path,
                          resume=False):
    char_counter = 0
    export_df = pd.DataFrame()
    # split data to chunks no longer then char_limit
    last_text_index, list_id_df, list_text_df = \
        chunk_texts_in_dataset(dataset, text_column, char_limit)
    # check if resuming previous translation
    if resume:
        start_at_chunk = resume
        export_df = pd.read_csv(output_path)
    else:
        start_at_chunk = 0
    for i in range(start_at_chunk, len(last_text_index)):
        try:
            # temp df
            trans_df = pd.DataFrame()
            # add origins to temp_df
            trans_df = trans_df.append(dataset.iloc[last_text_index[i] - len(list_id_df[i]):last_text_index[i]])
            # translate chunk of texts
            for x in list_text_df[i]:
                char_counter += len(x)
            print("Sent {} characters to translation so far. ".format(char_counter))
            chunk_translations = translate(list_text_df[i], src_lang, dest_lang)
            # add translations to temp df
            trans_df.insert(dataset.shape[1], translation_column, chunk_translations)
            # add translations to export df
            export_df = export_df.append(trans_df)
            # save export df to file
            export_csv = export_df.to_csv(output_path, index=False, header=True)
        except:  # if translation fails - prompt last chank to continue from
            last_text_index = last_text_index[i] - len(list_id_df[i])
            last_doc_id = 0 if i == 0 else list_id_df[(i - 1)][-1]
            print('Failed to translate at chunk {}. Translated {} indexes up to doc_id {}\n'
                  'Translated {} characters.Try again in 24 hours. Add resume= {} to run'.format
                  (i, last_text_index, last_doc_id, char_counter, i))
            sys.exit()
    last_text_index = last_text_index[i]
    last_doc_id = list_id_df[i][-1]
    print('Run complete with no Errors. Translated {} characters in {} chunks. Translated {} indexes up to doc_id {}.\n'
          'Translated from column "{}". Translated to column "{}". \n'
          'Translated from language "{}" to language "{}"'.format
          (char_counter, i, last_text_index, last_doc_id, text_column, translation_column, src_lang, dest_lang))
    return export_df


def parse_df(dataset, text_column, char_limit):  # parse df to list of dfs
    print('parsing dataset to parts not longer then {} chars in column {} to avoid exceeding daily translations limit'
          .format(char_limit, text_column))
    export_list = []
    last_text_index, list_id_df, list_text_df = \
        chunk_texts_in_dataset(dataset, text_column, char_limit)
    for i in range(0, len(last_text_index)):
        # temp df
        partial_df = pd.DataFrame()
        # first index of chunk = last index in chunk - number of instances in chunk (last_text_index[i]) - len(list_id_df[i])
        index_start_of_chunk = max(0, last_text_index[i] - len(list_id_df[i]))
        # add chunk to temp_df
        partial_df = partial_df.append(dataset.iloc[index_start_of_chunk:last_text_index[i]])
        export_list.append(partial_df)
    print('Divided dataset to {} parts.\n'
          'Each part has no more then {} chars in column {}.'.format
          (i + 1, char_limit, text_column))
    return export_list


def check_if_texts_to_translate_in_column_text(texts_to_translate_in_column, datafram, user_input=0):
    if texts_to_translate_in_column == 'text':
        part = parse_df(read_csv, texts_to_translate_in_column, 140000)
        chunk_to_translate = 0
        try:
            user_input = int(chunk_to_translate)
        except ValueError:
            print("This is not a valid number.")
            sys.exit(0)
        if 0 <= user_input <= (len(part) - 1):
            part_index = user_input
            return part, part_index
        else:
            raise ValueError("Value not in range of chunks created")
            sys.exit(0)
    # else we have a chunk of data already  translated at least once. no need to split again :
    else:
        part = [read_csv]
        user_input = 0
        return part, user_input


if __name__ == '__main__':
    # load dataset from csv to pandas DF
    read_csv = pd.read_csv(r'C:\Users\Roy\Documents\semi-supervised-text-classification\data\test_enron_en_de.csv')
    # create a list of dataset. Each dataset has no more characters then daily translation limit.
    texts_to_translate_in_column = 'de_trans'  # if 'text' insert chunk_number
    # if texts_to_translate_in_column == test => we are starting new translation:
    part, part_index = check_if_texts_to_translate_in_column_text(texts_to_translate_in_column, read_csv, user_input=0)
    # every day, run translation from src_lang to dest_lang on part[0]
    translations_df = create_translation_df(
        dataset=part[part_index],
        text_column=texts_to_translate_in_column,
        translation_column='de_bt',
        src_lang='de',
        dest_lang='en',
        char_limit=13000,
        output_path=r'C:\Users\Roy\Documents\semi-supervised-text-classification\data\test_enron_de_bt.csv',
        resume=False)

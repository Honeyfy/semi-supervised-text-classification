""" googletrans_augmentation.py create translations through googletrans library.
    Because of googletrans limitations on daily usage and maximum length per translation
    DataFrame is split to parts according to daily_char_limit of specified text column.
    The part chosen for translation is divided yet again to smaller bits (translation chunks) and sent for translation.
    translations are added to specified column and saved to new csv file.
    When translation chain ends successfully a merged data file of translations and it's origins is created.
    Usage: run_translation_chain(
        data_path = path to csv file with columns 'text' and 'doc_id'
        output_library_path = where csv files with translations will be saved
        translations_codes = default set to : ['en','fr','de','en'], (https://py-googletrans.readthedocs.io/en/latest/)
        start_chunk = what part of data will be translated)

     If translation fails during run:
     run again with code_position = last successful translation notice (['en','fr','de','en'] => [0 , 1 , 2 , 3 ])
     Translated data is saved in out_put_library_path
        GoodLuck """

import pandas as pd
from googletrans import Translator
import time
import sys
import numpy as np
import spacy

# global variables
daily_char_limit = 400000
trans_char_limit = 12800
sleep_length = 600
default_text_column = 'text'
nlp = spacy.load("en_core_web_sm")


# # for testing #
# def translate_test(origin, src, dest):  # for testing
#     origin_list = [origin] if isinstance(origin, str) else origin
#     translates_obj = origin_list
#     translates_list = list(map(lambda x: dest + x, translates_obj))
#     return translates_list


def translate(origin, src, dest):
    time.sleep(np.random.randint(12, 18))
    # check if origin is string or list. - if string return [string]
    origin_list = [origin] if isinstance(origin, str) else origin
    translator = Translator()
    translates_obj = translator.translate(origin_list, src=src, dest=dest)
    # create a list texts from the objects returned by translator
    translates_list = list(map(lambda x: x.text, translates_obj))
    return translates_list


def chunk_texts_in_dataset(dataset, text_column, char_limit):  # create lists with chunks of data from ds columns
    all_char_counter = 0  # counter for al characters in text
    char_counter = 0  # counter for characters in chunk
    doc_ids_chunk = []  # applying until chunk is full
    texts_chunk = []  # applying until chunk is full
    list_id_ds = []  # A list to keep chunks of doc_ids
    list_text_ds = []  # A list to keep chunks of texts
    # create a list of doc_ids from column
    doc_ids = dataset['document_id'].tolist()
    # create a list of texts from column
    texts = dataset[text_column].tolist()
    # check char length of longest text in list
    max_text_len = len(max(texts, key=len))
    if max_text_len >= char_limit:
        print(
            "Longest text has {} characters.\n"
            "Texts longer then char_limit of {} characters, are dropped from translation".format(
                max_text_len, char_limit))
    # start chunking
    # iterate over texts
    for texts_index in range(len(texts)):
        # if chunk has room for next text :count chars, add doc_id , add text
        if (char_counter + len(texts[texts_index])) <= char_limit:
            char_counter += len(texts[texts_index])
            doc_ids_chunk.append(doc_ids[texts_index])
            texts_chunk.append(texts[texts_index])
        # if text is longer then char_limit - drop it
        elif len(texts[texts_index]) >= char_limit:
            print('''document_id: {} at index {} is {} chars long and exceeds char limit.\n
             it is dropped'''.format(doc_ids[texts_index], texts_index,
                                     len(texts[texts_index])))
            pass
        else:  # reached char_limit - apply ds lists and restart chunk lists and char_counter
            list_id_ds.append(doc_ids_chunk)
            list_text_ds.append(texts_chunk)
            all_char_counter += char_counter
            char_counter = 0
            texts_chunk = []
            doc_ids_chunk = []
            char_counter += len(texts[texts_index])
            doc_ids_chunk.append(doc_ids[texts_index])
            texts_chunk.append(texts[texts_index])
    # finished iteration over all instances. apply final chunk to ds lists.
    all_char_counter += char_counter
    list_id_ds.append(doc_ids_chunk)
    list_text_ds.append(texts_chunk)
    print("dataset is {} characters long.".format(all_char_counter))
    return list_id_ds, list_text_ds


def create_translation_df(dataset, text_column, translation_column, src_lang, dest_lang, char_limit, output_path,
                          char_counter=0):
    export_df = pd.DataFrame()
    # split data to chunks no longer then char_limit
    list_id_df, list_text_df = chunk_texts_in_dataset(dataset, text_column, char_limit)
    start_at_chunk = 0
    for i in range(start_at_chunk, len(list_id_df)):
        try:
            # initialize a temp df with doc_ids
            temp_translations_df = pd.DataFrame(list_id_df[i], columns=['document_id'])
            # translate chunk of texts
            for x in list_text_df[i]:
                char_counter += len(x)
            print("Sent {} characters to translation so far. ".format(char_counter))
            chunk_translations = translate(list_text_df[i], src_lang, dest_lang)
            # add translations to temp df
            temp_translations_df.insert(temp_translations_df.shape[1], translation_column, chunk_translations)
            # add translations to export df
            export_df = export_df.append(temp_translations_df)
            # save export df to file on each run
            export_csv = export_df.to_csv(output_path, index=False, header=True)
        except:  # if translation fails - prompt last chank to continue from
            print('Failed to translate while trying to create {}. Translated {} characters'.format
                  (translation_column, char_counter))
            sys.exit()
    last_doc_id = list_id_df[i][-1]
    print('Run complete with no Errors. Translated {} characters in {} chunks. Translated up to doc_id {}.\n'
          'Translated from column "{}". Translated to column "{}". \n'
          'Translated from language "{}" to language "{}"'.format
          (char_counter, i, last_doc_id, text_column, translation_column, src_lang, dest_lang))
    return export_df, char_counter


def parse_ds(dataset, text_column, char_limit):  # parse ds to list of ds with specified number of characters
    export_list = []
    list_id_ds, list_text_ds = chunk_texts_in_dataset(dataset, text_column, char_limit)
    for i in range(0, len(list_id_ds)):
        # initialize temp ds
        partial_ds = pd.DataFrame()
        # append instances where document_id is in list_id_ds[i]
        partial_ds = partial_ds.append(dataset.loc[dataset['document_id'].isin(list_id_ds[i])])
        export_list.append(partial_ds)
    return export_list


def get_wanted_chunk_size(translation_codes):
    codes_count = len(translation_codes)
    return int(daily_char_limit / codes_count)


def get_ds_for_translation(data_path, wanted_chunk_size, start_chunk):
    dataset = pd.read_csv(data_path)
    ds_for_translation_chunks = parse_ds(dataset, default_text_column, wanted_chunk_size)
    ds_for_translation = ds_for_translation_chunks[start_chunk]
    return ds_for_translation, len(ds_for_translation_chunks) - 1


def merge_back_translations_with_origin(ds_translations, trans_path, ds_origin):
    ds_origins_of_translations = ds_origin.loc[ds_origin['document_id'].isin(ds_translations['document_id'])]
    ds_translations_with_origin_columns = ds_origins_of_translations.join(ds_translations.set_index('document_id'),
                                                                          on='document_id', how='inner', lsuffix='l')
    ds_translations_with_origin_columns['source'] = trans_path
    ds_translations_with_origin_columns['text'] = ds_translations_with_origin_columns[trans_path]
    ds_origin['source'] = 'origin'
    ds_translations_with_origin_columns.drop([trans_path], axis=1, inplace=True)
    merged_backtranslations = pd.concat((ds_origin, ds_translations_with_origin_columns))
    merged_backtranslations.reset_index(drop=True, inplace=True)
    return merged_backtranslations


def run_translation_chain(data_path, output_library_path, translation_codes, start_chunk, code_position=0):
    # get wanted chunk size to complete one translation chain in less then daily_char_limit
    wanted_chunk_size = get_wanted_chunk_size(translation_codes[1:])
    print("splitting data to chunks with length of {} characters".format(wanted_chunk_size))
    ds_for_translation, index_last_chunk = get_ds_for_translation(data_path, wanted_chunk_size, start_chunk)
    print("data was split to {} chunks. translating part {}".format(index_last_chunk, start_chunk))
    ds_origin = ds_for_translation.copy(deep=True)
    trans_path = default_text_column
    char_counter = 0
    # prepare variables and data to continue an incomplete previous translation chain
    if code_position:
        for i, trans_code in enumerate(translation_codes[1:code_position + 1], 1):
            new_trans_path = trans_path + "_" + trans_code
            output_path = output_library_path + new_trans_path + str(start_chunk) + ".csv"
            ds_for_translation = pd.read_csv(output_path)
            trans_path = new_trans_path
    # start translation chain
    for i, trans_code in enumerate(translation_codes[code_position + 1:], code_position + 1):
        new_trans_path = trans_path + "_" + trans_code
        output_path = output_library_path + new_trans_path + str(start_chunk) + ".csv"
        generated_translations, char_counter = create_translation_df(ds_for_translation, trans_path,
                                                                     new_trans_path, translation_codes[i - 1],
                                                                     trans_code, trans_char_limit, output_path,
                                                                     char_counter)
        ds_for_translation = generated_translations
        trans_path = new_trans_path
        # sleep between backtranslations unless last translation
        if i != len(translation_codes[code_position + 1:]):
            print("Finished translation to: ", trans_code, ". Sleep for : ", sleep_length, "seconds.")
            time.sleep(sleep_length)
    # create a dataset with translations and it's origins
    merged_back_translated_data_set = merge_back_translations_with_origin(ds_for_translation, trans_path, ds_origin)
    export_csv = merged_back_translated_data_set.to_csv(
        output_library_path + trans_path + str(start_chunk) + "_merged_with_origin.csv", index=False, header=True)
    print("Index of last chunk is {}. Translation of chunk {} ended successfully".format(index_last_chunk, start_chunk))
    return ds_for_translation, merged_back_translated_data_set


if __name__ == '__main__':
    translated_texts, merged_translations = run_translation_chain(
        data_path=r'C:\Users\Roy\Documents\semi-supervised-text-classification\data\tests\real_aug\data.csv',
        output_library_path=r'C:\Users\Roy\Documents\semi-supervised-text-classification\data\tests\real_aug\\',
        translation_codes=['en', 'fr', 'de', 'en'],
        start_chunk=0,
        code_position=0)

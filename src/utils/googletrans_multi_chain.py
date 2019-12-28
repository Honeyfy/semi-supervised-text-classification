""" googletrans_multi_chain.py create multiple backtranslations through googletrans library.
    Because of googletrans limitations on daily usage and maximum length per translation
    DataFrame is split to parts according to daily_char_limit of specified text column.
    The part chosen for translation is divided yet again to smaller bits (translation chunks) and sent for translation.
    translations are added to specified column and saved to new csv file.
    When translation chain ends successfully a merged data file of translations and it's origins is created.
    Usage: run_translation_chain(
        data_path = path to csv file with columns 'text' and 'doc_id'
        output_library_path = where csv files with translations will be saved
        translations_codes = default set to : [['en','fr','de','en'], ['en','vi','en']],
                            (https://py-googletrans.readthedocs.io/en/latest/)
        start_chunk = what part of data will be translated
        code_position=[0, 0] from where to resume translation of current chunk

     If translation fails during run:
     run again with code_position = last successful translation notice ([['en','fr','de','en'], ['en','vi','en']] => )
                                                                        [0,0] [0,1] [0,2] [0,3]  [1,0] [1,1][1,2]
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
trans_char_limit = 12700
sleep_length = 600
nlp = spacy.load("en_core_web_sm")
separator = '='


# # for testing #
# def translate(origin, src, dest):  # for testing
#     origin_list = [origin] if isinstance(origin, str) else origin
#     translates_obj = origin_list
#     translates_list = list(map(lambda x: dest + x, translates_obj))
#     if dest == 'F':
#         pass
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
    print("Full dataset is {} characters long.".format(all_char_counter))
    return list_id_ds, list_text_ds


def create_translation_df(dataset, text_column, translation_column, src_lang, dest_lang, char_limit, output_path,
                          translation_counter):
    export_df = pd.DataFrame()
    start_at_chunk = 0
    # split data to chunks no longer then char_limit
    list_id_df, list_text_df = chunk_texts_in_dataset(dataset, text_column, char_limit)
    for i in range(start_at_chunk, len(list_id_df)):
        # translate chunk of texts
        for x in list_text_df[i]:
            translation_counter += len(x)
        print("Sent {} characters for translation so far. ".format(translation_counter))
        chunk_translations = translate(list_text_df[i], src_lang, dest_lang)
        # initialize a temp df with doc_ids
        temp_translations_df = pd.DataFrame(list_id_df[i], columns=['document_id'])
        # add translations to temp df
        temp_translations_df.insert(temp_translations_df.shape[1], translation_column, chunk_translations)
        # add translations to export df
        export_df = export_df.append(temp_translations_df, sort=True)
        # save export df to file on each run
        export_csv = export_df.to_csv(output_path, index=False, header=True)
    return export_df, translation_counter


def parse_ds(dataset, text_column, char_limit):  # parse ds to list of ds with specified number of characters
    export_list = []
    list_id_ds, list_text_ds = chunk_texts_in_dataset(dataset, text_column, char_limit)
    for i in range(0, len(list_id_ds)):
        # initialize temp ds
        partial_ds = pd.DataFrame()
        # append instances where document_id is in list_id_ds[i]
        partial_ds = partial_ds.append(dataset.loc[dataset['document_id'].isin(list_id_ds[i])], sort=True)
        export_list.append(partial_ds)
    return export_list


def get_wanted_chunk_size(translation_codes):
    codes_count = 0
    for translation_chain in translation_codes:
        codes_count += len(translation_chain[1:])
    return int(daily_char_limit / codes_count)


def get_ds_for_translation(default_text_column, data_path, wanted_chunk_size, start_chunk):
    dataset = pd.read_csv(data_path)
    ds_for_translation_chunks = parse_ds(dataset, default_text_column, wanted_chunk_size)
    ds_for_translation = ds_for_translation_chunks[start_chunk]
    return ds_for_translation, len(ds_for_translation_chunks) - 1


# merge translations with original ds
def merge_back_translations_with_origin(ds_translations, trans_path, ds_origin, include_origins=False):
    if include_origins:
        ds_origin['source'] = 'origin'
        merged_backtranslations = pd.concat((ds_origin, ds_translations), sort=True)
        merged_backtranslations.reset_index(drop=True, inplace=True)
        return merged_backtranslations
    ds_origins_of_translations = ds_origin.loc[ds_origin['document_id'].isin(ds_translations['document_id'])]
    ds_translations_with_origin_columns = ds_origins_of_translations.join(ds_translations.set_index('document_id'),
                                                                          on='document_id', how='inner', lsuffix='l')
    ds_translations_with_origin_columns['source'] = trans_path
    ds_translations_with_origin_columns['text'] = ds_translations_with_origin_columns[trans_path]
    ds_origin['source'] = 'origin'
    ds_translations_with_origin_columns.drop([trans_path], axis=1, inplace=True)
    return ds_translations_with_origin_columns


def run_translation_chains(default_text_column, data_path, output_library_path, translation_codes_in, start_chunk,
                           code_position):
    # get wanted chunk size to complete one translation chain in less then daily_char_limit
    wanted_chunk_size = get_wanted_chunk_size(translation_codes_in)
    print("splitting data to chunks with length of {} characters".format(wanted_chunk_size))
    origins_chunk_for_translation, index_last_chunk = get_ds_for_translation(default_text_column, data_path,
                                                                             wanted_chunk_size,
                                                                             start_chunk)
    print("data was split to chunks: {}. translating chunk {}".format(np.arange(0, index_last_chunk), start_chunk))
    ds_for_translation = origins_chunk_for_translation.copy(deep=True)
    trans_path = default_text_column
    translation_counter = 0
    all_back_translations = pd.DataFrame()
    all_translations = []
    # prepare variables and data to continue an incomplete previous translation chain
    if code_position[0]:
        try:
            all_back_translations = load_chunk_bt_from_previous_translation_chains(default_text_column,
                                                                                   output_library_path,
                                                                                   translation_codes_in, start_chunk,
                                                                                   code_position)
        except:
            print(separator * 6,
                  "\n Failed to load previous translatoins. \n Check bt_with_origins{}.csv  after run is complete "
                  "to see which translations are missing from it\n".format(start_chunk), separator * 6)
            time.sleep(10)
    if code_position[0] or code_position[1]:
        translation_codes = translation_codes_in[code_position[0]:]
        for i, trans_code in enumerate(translation_codes[0][1:code_position[1] + 1], 1):
            new_trans_path = trans_path + "_" + trans_code
            output_path = output_library_path + new_trans_path + str(start_chunk) + ".csv"
            ds_for_translation = pd.read_csv(output_path)
            trans_path = new_trans_path
    else:
        translation_codes = translation_codes_in
    # start translation chain
    for chain_index, translation_chain in enumerate(translation_codes):
        translations_list, back_translations_data_set, translation_counter \
            = translate_chain(ds_for_translation, output_library_path, trans_path, translation_chain, start_chunk,
                              translation_counter, code_position=code_position[1])
        all_back_translations = all_back_translations.append(back_translations_data_set, sort=True)
        all_translations.append(translations_list)
        print("Finished translation chain : ", translation_chain, ". Chain index number: ",
              chain_index + code_position[0])
        code_position[1] = 0
        trans_path = default_text_column
        ds_for_translation = origins_chunk_for_translation.copy(deep=True)
    print("Translation of chunk {} ended successfully. Index of last chunk in data is {} . \n"
          "Translation chains in current run were : {} ".format(start_chunk, index_last_chunk, translation_codes_in))
    all_back_translations_with_origins = merge_back_translations_with_origin(all_back_translations, trans_path,
                                                                             origins_chunk_for_translation,
                                                                             include_origins=True)
    export_csv = all_back_translations_with_origins.to_csv(
        output_library_path + "bt_with_origins_chunk_" + str(start_chunk) + ".csv", index=False, header=True)
    return all_translations, all_back_translations_with_origins


def load_chunk_bt_from_previous_translation_chains(default_text_column, output_library_path,
                                                   translation_codes, start_chunk, code_position):
    prev_back_translations = pd.DataFrame()
    for trans_chain in translation_codes[:code_position[0]]:
        bt_file_name = output_library_path[:-1] + default_text_column
        for translation_code in trans_chain[1:]:
            bt_file_name += "_" + translation_code
        bt_file_name += str(start_chunk) + "_back_translations_data_set.csv"
        back_translations = pd.read_csv(bt_file_name)
        prev_back_translations = prev_back_translations.append(back_translations, sort=True)
    return prev_back_translations


def translate_chain(ds_origin, output_library_path, trans_path, translation_chain, start_chunk, translation_counter,
                    code_position):
    ds_for_translation = ds_origin.copy(deep=True)
    for i, trans_code in enumerate(translation_chain[code_position + 1:], code_position + 1):
        new_trans_path = trans_path + "_" + trans_code
        output_path = output_library_path + new_trans_path + str(start_chunk) + ".csv"
        try:
            generated_translations, translation_counter \
                = create_translation_df(ds_for_translation, trans_path, new_trans_path, translation_chain[i - 1],
                                        trans_code, trans_char_limit, output_path, translation_counter)
        except:
            print("Failed to translate while translating chunk {} \n"
                  "Resume on Translation chain {} to code {} ".format(start_chunk, translation_chain,
                                                                      translation_chain[max(0, (i - 1))]))
            sys.exit(1)
        ds_for_translation = generated_translations
        trans_path = new_trans_path
        # sleep between backtranslations unless last translation
        if i != len(translation_chain[code_position + 1:]):
            print("\n", separator * 10, "Finished translation to: ", trans_code, ". Sleep for : ", sleep_length,
                  "seconds.")
            time.sleep(sleep_length)
        print("\n", separator * 10, "\n")
    # create a dataset of back_translations and original dataset columns
    back_translations_data_set = merge_back_translations_with_origin(ds_for_translation, trans_path, ds_origin,
                                                                     include_origins=False)
    export_csv = back_translations_data_set.to_csv(
        output_library_path + trans_path + str(start_chunk) + "_back_translations_data_set.csv", index=False,
        header=True)
    return ds_for_translation, back_translations_data_set, translation_counter


if __name__ == '__main__':
    data_path = r'C:\Users\Roy\Documents\semi-supervised-text-classification\data\pres\\'
    translated_texts, merged_translations = run_translation_chains(
        default_text_column='text',
        data_path=data_path+'origins_of_augmented.csv',
        output_library_path=r'C:\Users\Roy\Documents\semi-supervised-text-classification\data\tests\data0\\',
        translation_codes_in=[['en', 'zh-cn', 'en']],
        start_chunk=0,
        code_position=[0, 0]
    )

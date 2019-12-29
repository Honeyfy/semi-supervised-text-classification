"""
Data augmentation with synonyms. A number of adjectives in each text are replaced with some synonyms.
The synonyms are collected by scraping website www.thesaurus.com.
Written by Shira Weissman
"""


import spacy
import json
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import itertools
import re


class SynonymsDataAugmentation():
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    def __init__(self, data, full_synonyms_dict_file_path, words_to_exclude_file_path,
                 only_synonyms_already_in_text_file_path,
                 synonyms_of_each_other_dict_file_path,
                 synonyms_augmented_data_file_path,
                 max_words_to_replace=3, max_synonyms_to_use=2):
        self.nlp = spacy.load('en_core_web_lg')
        self.data = data
        self.full_synonyms_dict_file_path = full_synonyms_dict_file_path
        self.words_to_exclude_file_path = words_to_exclude_file_path
        self.only_synonyms_already_in_text_file_path = only_synonyms_already_in_text_file_path
        self.synonyms_of_each_other_dict_file_path = synonyms_of_each_other_dict_file_path
        self.synonyms_augmented_data_file_path = synonyms_augmented_data_file_path
        self.full_synonyms_dict = {}
        self.words_to_exclude_dict = {'words_to_exclude': []}
        self.only_synonyms_already_in_text_dict = {}
        self.synonyms_of_each_other_dict = {}
        self.synonyms_augmented_data_df = None
        self.max_words_to_replace = max_words_to_replace
        self.max_synonyms_to_use = max_synonyms_to_use
        self.candidates_for_replacing = None

    def load_available_dictionaries(self):
        print("Loading available dictionaries..")
        try:
            with open(self.full_synonyms_dict_file_path, 'r') as f:
                self.full_synonyms_dict = json.loads(f.read())
        except:
            self.full_synonyms_dict = {}

        try:
            with open(self.words_to_exclude_file_path, 'r') as f:
                self.words_to_exclude_dict = json.loads(f.read())
        except:
            self.words_to_exclude_dict = {'words_to_exclude': []}

        try:
            with open(self.only_synonyms_already_in_text_file_path, 'r') as f:
                self.only_synonyms_already_in_text_dict = json.loads(f.read())
        except:
            self.only_synonyms_already_in_text_dict = {}

        try:
            with open(self.synonyms_of_each_other_dict_file_path, 'r') as f:
                self.synonyms_of_each_other_dict = json.loads(f.read())
        except:
            self.synonyms_of_each_other_dict = {}

    def synonyms(self, term):
        response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
        soup = BeautifulSoup(response.text, "html")
        soup2 = str(soup).split('window.INITIAL_STATE = ')[1]
        soup3 = soup2.split('"synonyms":')[1]
        soup_synonyms = soup3.split('"antonyms":')[0][:-1]
        soup_antonyms = soup3.split('"antonyms":')[1]
        soup_antonyms = soup_antonyms.split(']')[0] + ']'
        synonyms = pd.DataFrame(json.loads(soup_synonyms))
        antonyms = pd.DataFrame(json.loads(soup_antonyms)) #This dataframe which is not used in this code, contains antonyms of the term.
        max_similarity = synonyms.loc[0, 'similarity']
        synonyms_top_list = synonyms[synonyms['similarity'] == max_similarity]['term'].to_list()
        return synonyms_top_list

    def replace_multiple_words_with_synonyms(self, sentence, list_of_words_to_replace, synonyms_dict):
        # List of words from the given sentence is replaced by synonyms from the dictionary.
        # The list of words and lists of synonyms are shorten up to the parameters: max_words_to_replace, max_synonyms_to_use
        # This function return a list of augmented sentences, which includes the original sentence.
        if len(list_of_words_to_replace) == 0:
            return [sentence]
        synonyms_dict_for_words_to_replace = {}
        if len(set(list_of_words_to_replace)) > self.max_words_to_replace:
            list_of_words_to_replace = list(set(list_of_words_to_replace))[:self.max_words_to_replace]
        for word in list_of_words_to_replace:
            synonyms_dict_for_words_to_replace[word] = synonyms_dict[word]
            if len(synonyms_dict_for_words_to_replace[word]) > self.max_synonyms_to_use:
                synonyms_dict_for_words_to_replace[word] = synonyms_dict_for_words_to_replace[word][
                                                           :self.max_synonyms_to_use]

        sentence_list_of_words = sentence.split(' ')
        words_to_replace_with_index_list = [(i, word) for i, word in enumerate(sentence_list_of_words) if
                                            word in list_of_words_to_replace]
        list_of_synonyms_list = [synonyms_dict_for_words_to_replace[word_tuple[1]] + [word_tuple[1]] for
                                 word_tuple in words_to_replace_with_index_list]

        synonyms_permutations = list(itertools.product(*list_of_synonyms_list))
        vowels = re.compile('[aeiou]')
        augmented_sentences_list = []
        for synonyms_list in synonyms_permutations:
            new_sent_list = sentence_list_of_words
            for i, word_tuple in enumerate(words_to_replace_with_index_list):
                new_sent_list[word_tuple[0]] = synonyms_list[i]
                if new_sent_list[word_tuple[0] - 1] == 'a' or new_sent_list[word_tuple[0] - 1] == 'an':
                    if (re.match(vowels, synonyms_list[i][0]) == None):
                        new_sent_list[word_tuple[0] - 1] = 'a'
                    else:
                        new_sent_list[word_tuple[0] - 1] = 'an'
            new_sent = ' '.join(new_sent_list)
            augmented_sentences_list.append(new_sent)
        return augmented_sentences_list


    def pos_from_sentence(self, sentence, pos='ADJ'):
        # Extracting all the words which are the given part of speech from the sentence.
        # The default is adjective.
        words_to_exclude = self.words_to_exclude_dict['words_to_exclude']
        if len(words_to_exclude) > 0:
            exclude_template = re.compile('.*[\d\W]|.*' + '|'.join([word for word in words_to_exclude]))
        else:
            exclude_template = re.compile('.*[\d\W].*')
        nlp_sent = self.nlp(sentence)
        pos_word_list = [word.text for word in nlp_sent if word.pos_ == pos]
        final_list = [word for word in pos_word_list if re.match(exclude_template, word) == None]
        return final_list

    def word_in_sentence_exist_in_dict(self, sentence, synonyms_dict):
        nlp_sent = self.nlp(sentence)
        word_list = [word.text for word in nlp_sent if word.text in list(synonyms_dict.keys())]
        return word_list

    def synonyms_augmented_data(self, synonyms_dict):
        print("Generating augmented data..")
        data = self.data
        data['text'] = data.apply(lambda x: x['text'].lower(), axis=1)
        data['words_to_replace'] = data.apply(lambda x: [word for word in
                                                         self.word_in_sentence_exist_in_dict(x['text'], synonyms_dict)
                                                         if word in x['candidates_for_replacing']], axis=1)
        print("done with words to replace")
        data['augmented text'] = data.apply(lambda x: self.replace_multiple_words_with_synonyms(x['text'],
                                                                                                x['words_to_replace'],
                                                                                                synonyms_dict), axis=1)
        print("done with creating augmented_data")
        augmented_data = data.explode('augmented text').reset_index(drop=True)
        augmented_data.dropna(subset=['augmented text'], inplace=True)
        with open(self.synonyms_augmented_data_file_path, 'w') as f:
            augmented_data.to_csv(f, index=False)
        self.synonyms_augmented_data_df = augmented_data
        return augmented_data

    def get_words_candidates_for_replacing_from_data(self):
        # Applying the function pos_from_sentence to all texts in the data
        print("Extracting candidate words for replacing from data..")
        data = self.data
        data['candidates_for_replacing'] = data.apply(
            lambda x: self.pos_from_sentence(x['text'].lower()), axis=1)
        candidates_for_replacing = np.unique(data['candidates_for_replacing'].sum())
        data.drop(['candidates_for_replacing'], axis=1)
        self.candidates_for_replacing = candidates_for_replacing
        print("Number of candidates for replacing: {}".format(len(candidates_for_replacing)))
        print()
        return candidates_for_replacing

    def from_website_to_synonyms_dict(self, list_of_terms):
        # Scraping the list of terms from www.thesaurus.com
        print("Creating full dictionary for all candidate words for replacing..")
        print()
        synonyms_dict = self.full_synonyms_dict
        words_to_exclude_dict = self.words_to_exclude_dict

        for term in list_of_terms:
            if (term not in synonyms_dict) and (term not in words_to_exclude_dict['words_to_exclude']):
                try:
                    synonyms_top_list = self.synonyms(term)
                    # print("Scrapping {}".format(term))
                    synonyms_dict[term] = synonyms_top_list
                except:
                    words_to_exclude_dict['words_to_exclude'].append(term)
                    # print("The word {} is excluded.".format(term))
            # else:
            #     print("{} already in dictionary or in excluded words list.".format(term))

        self.full_synonyms_dict = synonyms_dict
        with open(self.full_synonyms_dict_file_path, 'w') as f:
            json.dump(synonyms_dict, f)

        words_to_exclude_dict['words_to_exclude'] = sorted(words_to_exclude_dict['words_to_exclude'])
        self.words_to_exclude_dict = words_to_exclude_dict
        with open(self.words_to_exclude_file_path, 'w') as f:
            json.dump(words_to_exclude_dict, f)

        return synonyms_dict, words_to_exclude_dict['words_to_exclude']

    def get_synonyms_also_terms_in_dict(self):
        # Delete synonyms which are not terms in the dictionary.
        # Delete terms which were left with an empty synonyms list.
        print("Reducing synonyms list to only synonyms which are also terms in the dictionary..")
        print()
        words_to_exclude_dict = self.words_to_exclude_dict
        synonyms_dict = self.full_synonyms_dict

        dict_of_synonyms_from_terms_list_only = {}
        terms_list = list(synonyms_dict.keys())
        for term in terms_list:
            dict_of_synonyms_from_terms_list_only[term] = [syn for syn in synonyms_dict[term] if syn in terms_list]
            print("Term: {}, synonyms list is shorten from {} to {}".format(term, len(synonyms_dict[term]), len(
                dict_of_synonyms_from_terms_list_only[term])))
            if len(dict_of_synonyms_from_terms_list_only[term]) == 0:
                del dict_of_synonyms_from_terms_list_only[term]
                if term not in words_to_exclude_dict['words_to_exclude']:
                    words_to_exclude_dict['words_to_exclude'].append(term)

        print()
        print("Number of terms in dictionary with synonyms only from text: {}".format(
            len(list(dict_of_synonyms_from_terms_list_only.keys()))))
        print("Number of terms which were removed: {}".format(
            len(terms_list) - len(list(dict_of_synonyms_from_terms_list_only.keys()))))

        self.only_synonyms_already_in_text_dict = dict_of_synonyms_from_terms_list_only
        with open(self.only_synonyms_already_in_text_file_path, 'w') as f:
            json.dump(dict_of_synonyms_from_terms_list_only, f)

        self.words_to_exclude_dict = words_to_exclude_dict
        with open(self.words_to_exclude_file_path, 'w') as f:
            json.dump(words_to_exclude_dict, f)
        return dict_of_synonyms_from_terms_list_only, words_to_exclude_dict['words_to_exclude']

    def get_synonyms_only_synonyms_of_each_other(self):
        print("Reducing dictionary to terms which are synonyms of each other..")
        synonyms_dict = self.only_synonyms_already_in_text_dict

        terms_list = list(synonyms_dict.keys())
        synonyms_of_each_other_dict = {}

        for term in terms_list:
            synonyms_of_each_other_dict[term] = []
            for word in synonyms_dict[term]:
                if word in terms_list:
                    if term in synonyms_dict[word]:
                        synonyms_of_each_other_dict[term].append(word)
            print("Term: {}, synonyms list is shorten from {} to {}".format(term, len(synonyms_dict[term]),
                                                                            len(synonyms_of_each_other_dict[term])))

            if len(synonyms_of_each_other_dict[term]) == 0:
                del synonyms_of_each_other_dict[term]
                if term not in self.words_to_exclude_dict['words_to_exclude']:
                    self.words_to_exclude_dict['words_to_exclude'].append(term)

        print()
        print("Number of terms in dictionary which are synonyms in both directions: {}".format(
            len(list(synonyms_of_each_other_dict.keys()))))
        print("Number of terms which were removed: {}".format(
            len(terms_list) - len(list(synonyms_of_each_other_dict.keys()))))

        self.synonyms_of_each_other_dict = synonyms_of_each_other_dict
        with open(self.synonyms_of_each_other_dict_file_path, 'w') as f:
            json.dump(synonyms_of_each_other_dict, f)
        return synonyms_of_each_other_dict, self.words_to_exclude_dict['words_to_exclude']

    def get_words_to_replace_exist_in_synonyms_dict(self, synonyms_dict):
        # 'words_to_replace' is the list of final chosen words for creating the augmented sentences.
        data = self.data
        data['words_to_replace'] = data.apply(
            lambda x: [word for word in self.pos_from_sentence(x['text'].lower()) if
                       word in list(synonyms_dict.keys())], axis=1)
        return data

    def run_all_data_augmentation_process(self):

        self.load_available_dictionaries()

        candidates_for_replacing = self.get_words_candidates_for_replacing_from_data()

        synonyms_dict, words_to_exclude_list = self.from_website_to_synonyms_dict(candidates_for_replacing)
        print("Number of words to exclude: {}".format(len(words_to_exclude_list)))
        print()
        dict_of_synonyms_from_terms_list_only, words_to_exclude_list = self.get_synonyms_also_terms_in_dict()

        synonyms_of_each_other_dict, words_to_exclude_list = self.get_synonyms_only_synonyms_of_each_other()
        print("Number of total words to exclude: {}".format(len(words_to_exclude_list)))
        print()

        augmented_data = self.synonyms_augmented_data(synonyms_of_each_other_dict)



if __name__ == '__main__':
    file_path = r'C:\develop\code\semi-supervised-text-classification\data\ml_input.csv'
    with open(file_path, 'r') as f:
        data = pd.read_csv(f)
    data = data.drop(['Unnamed: 0'], axis=1)

    full_synonyms_dict_file_path = r'C:\develop\code\semi-supervised-text-classification\data\synonyms_dict.txt'
    words_to_exclude_file_path = r'C:\develop\code\semi-supervised-text-classification\data\words_to_exclude.txt'
    only_synonyms_already_in_text_file_path = r'C:\develop\code\semi-supervised-text-classification\data\only_synonyms_already_in_text_dict.txt'
    synonyms_of_each_other_dict_file_path = r'C:\develop\code\semi-supervised-text-classification\data\synonyms_of_each_other_dict.txt'
    synonyms_augmented_data_file_path = r'C:\develop\code\semi-supervised-text-classification\data\synonyms_augmented_data.csv'

    max_words_to_replace = 3
    max_synonyms_to_use = 2

    sda = SynonymsDataAugmentation(data,
                                   full_synonyms_dict_file_path,
                                   words_to_exclude_file_path,
                                   only_synonyms_already_in_text_file_path,
                                   synonyms_of_each_other_dict_file_path,
                                   synonyms_augmented_data_file_path,
                                   max_words_to_replace,
                                   max_synonyms_to_use)

    sda.run_all_data_augmentation_process()

'''
Compare editdistance between original and augmented text
'''
# TODO 1.check edit distance - checked - 2.combine different senteces 3.create clean backtranslation function
import editdistance
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import pickle
from nltk import sent_tokenize

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import os
from src.models.text_classifier import run_model_on_file
from src.models.text_classifier import TextClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def run_model_on_texts(file_path, project_id):
    print("run model on full texts to create classifier")
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    input_file = os.path.join(data_path, file_path)
    output_file = os.path.join(data_path, 'output.csv')
    result = run_model_on_file(
        input_filename=input_file,
        output_filename=output_file,
        project_id=project_id,
        user_id=None,
        label_id=None,
        run_on_entire_dataset=False)


def print_evaluation_scores(y, y_pred):
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print("# # # # # # # # # MODEL SCORES ON EXTERNAL TEST DATA # # # # # # # # # # #\n"
          " precision :{}   recall :{}    f1 :{}   on Test file".format(precision, recall, f1))


def lower_remove_special_chars_and_split(in_string):  # remove special chars from string
    import re
    if in_string:
        lower_string = in_string.lower()
        # replace special chars with " "
        clean_string = re.sub('[!#$-%:;(_)?.,]', ' ', lower_string)
        # split stirng on " "
        clean_string_split = clean_string.split(" ")
        # return list without " " items
        return_string = [value for value in clean_string_split if value != ""]
        return return_string
    else:
        return in_string

def get_sentence_edit_distance(sent1, sent2):
    sent1_list_of_words = lower_remove_special_chars_and_split(sent1)
    sent2_list_of_words = lower_remove_special_chars_and_split(sent2)
    return editdistance.eval(sent1_list_of_words, sent2_list_of_words)

def load_model_from_file(model_id):
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    model_file = 'results\ml_model_' + str(model_id) + '.pickle'
    model_path = data_path + '\\' + model_file
    clf = TextClassifier.load(model_path)
    return clf


def get_sentences_editdistance_list(text1, text2):  # takes 2 texts, sent toknize eace text, compare the sentences.
    distances_list = []
    text1_sentences = sent_tokenize(text1)
    text2_sentences = sent_tokenize(text2)
    number_of_sentences = min(len(text1_sentences), len(text2_sentences))
    for sent_index in range(number_of_sentences):
        distances_list.append(get_sentence_edit_distance(text1_sentences[sent_index], text2_sentences[sent_index]))
    if len(text1_sentences) != len(text2_sentences):
        # print("different number of sentences in text1: {} ,"
        # " and text2: {} is different input -1 to list".format(len(text1_sentences), len(text2_sentences)))
        return -1
    return distances_list


def compare_augmentation_df(data_frame, source1, source2):
    """ input: DF with column[document id, text, source] source contain labels for augmentations.
    output: DF with editdistance between their text concatenated on document_id """
    # split dataframe on source1
    comparison_df1 = pd.DataFrame()
    comparison_df1[['document_id', source1]] = data_frame[['document_id', 'text']].loc[data_frame['source'] == source1]
    # split dataframe on source2
    comparison_df2 = pd.DataFrame()
    comparison_df2[['document_id', source2]] = data_frame[['document_id', 'text']].loc[data_frame['source'] == source2]
    # concatenate on document_id
    comparison_df = pd.concat([comparison_df1.set_index('document_id'), comparison_df2.set_index('document_id')],
                              axis=1, join='inner').reset_index()
    # compute editdistance
    comparison_df['editdistance'] = comparison_df.apply(
        lambda x: get_sentences_editdistance_list(x[source1], x[source2]), axis=1)
    return comparison_df


def get_df_avg_distance(data_frame, source1, source2):
    comparison_df = compare_augmentation_df(data_frame, source1, source2)
    return get_avg_distance(comparison_df)

def get_avg_distance(comparison_df):  # input: comparison df, output: average sentence editdistance
    valid_df = comparison_df.loc[comparison_df['editdistance'] != -1]
    print("droped {} invalid comparisons".format(len(comparison_df) - len(valid_df)))
    valid_df['distance_sum'] = valid_df['editdistance'].apply(lambda x: sum(x) / len(x))
    return valid_df['distance_sum'].sum() / len(valid_df)


if __name__ == '__main__':
    # set data path
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    # load augmented data set
    augmented_data_path = os.path.join(data_path, 'ml_augmented.csv')
    augmented_data_set = pd.read_csv(augmented_data_path)
    # remove first line of file -  it contains comments
    augmented_data_set = augmented_data_set[1:]
    # set column names
    augmented_data_set.columns = augmented_data_set.iloc[0]
    # now remove line containing column names from data
    augmented_data_set = augmented_data_set[1:]
    print(" fr ", get_df_avg_distance(augmented_data_set,'origin','bt fr'))
    print(" frgr ", get_df_avg_distance(augmented_data_set,'origin','bt fr-gr'))
    print(" he ", get_df_avg_distance(augmented_data_set, 'origin', 'bt he'))
    print(" vi ", get_df_avg_distance(augmented_data_set, 'origin', 'bt vi'))


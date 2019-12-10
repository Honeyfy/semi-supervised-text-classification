import pandas as pd
import numpy as np
import os
from src.models.text_classifier import TextClassifier


def load_data(filename):
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    input_file = os.path.join(data_path, filename)
    with open(input_file, 'r') as f:
        data = pd.read_csv(f)
    return data

def textDf_2_tokens(data):
    pattern = '\n\n'
    data['text'] = data['text'].str.split(pattern, expand=False)
    paragraphs = data.explode('text').reset_index(drop=True)
    paragraphs.dropna(subset=['text'], inplace=True)
    return paragraphs

def predict_paragraphs(paragraphs, model_id):
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    model_file = 'results\ml_model_' + str(model_id) + '.pickle'
    model_path = data_path + '\\' + model_file
    clf = TextClassifier.load(model_path)
    X = clf.pre_process(paragraphs, fit=False)
    paragraphs_pred = clf.get_prediction_df(X)
    paragraphs_pred.reset_index(drop=True, inplace=True)
    paragraphs.reset_index(drop=True, inplace=True)
    paragraphs_with_prediction = pd.concat([paragraphs, paragraphs_pred], axis=1)
    paragraphs_with_prediction['prediction'] = paragraphs_with_prediction['prediction'].astype(int)
    return paragraphs_with_prediction


def get_pargarphs_predicted_1_with_high_conf(dataframe, confidence):
    predictions_from_1_docs = dataframe.loc[dataframe['label_id'] == 1]
    predictions_of_1 = predictions_from_1_docs.loc[predictions_from_1_docs['prediction'] == '1']
    predictions_of_1_high_conf = predictions_of_1.loc[predictions_of_1['confidence'] > confidence]
    return predictions_of_1_high_conf

def get_conf_labels(model_id, par_text_df, min_confidence, user_id, include_zeros=False):
    one_string = ''       #TODO what hapens when b<1?
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    model_file = 'results\ml_model_' + str(model_id) + '.pickle'
    model_path = data_path + '\\' + model_file
    clf = TextClassifier.load(model_path)
    pred_df = par_text_df.copy()
    # text pre_preprocess
    X_preprocess = clf.pre_process(pred_df,fit = False)
    X_predictions_df = clf.get_prediction_df(X_preprocess)
    #X_predictions_df['text'] = pred_df['processed_text']
    X_predictions_df.insert(0, 'text', pred_df['processed_text'].values)
    X_predictions_df.insert(0, 'document_id', pred_df['document_id'].values)
    # get conf labels
    a = len(X_predictions_df)
    X_predictions_df['prediction'] = X_predictions_df['prediction'].astype('int')
    new_labels_df = X_predictions_df.loc[(X_predictions_df['confidence'] > min_confidence) & (X_predictions_df['prediction'] == 1) ]
    b= len(new_labels_df)

    if len(new_labels_df)>0 :
        new_labels_df.rename(columns={"prediction": "label_id"}, inplace=True)
        new_labels_df.drop('confidence', axis=1, inplace=True)
        new_labels_df['user_id'] = user_id
        new_labels_df = new_labels_df[['document_id', 'label_id', 'text', 'user_id']]


    return new_labels_df

def get_train_set_for_1_labels(full_texts_df,pargarphs_predicted_1_with_high_conf):
    labeled_1_full_texts_not_in_par_pred = full_texts_df.loc[~full_texts_df['document_id'].isin(pargarphs_predicted_1_with_high_conf['document_id'])]
    train_set_for_1_labels = pd.concat((labeled_1_full_texts_not_in_par_pred,pargarphs_predicted_1_with_high_conf), axis=0)
    return train_set_for_1_labels

def get_paragraphs_0_labels(pargarphs):
    pargarphs0 = pargarphs[pargarphs['label_id']==0]
    return pargarphs0

def get_paragraphs_1_labels(pargarphs):
    pargarphs1 = pargarphs[pargarphs['label_id']==1]
    return pargarphs1

def create_new_dataset(predictions_of_1_high_conf, train_set_for_1_labels, pargarphs0):
    # predictions_of_1_high_conf = predictions_of_1_high_conf.drop(['prediction', 'confidence'], axis=1)
    predictions_of_1_high_conf.reset_index(drop=True, inplace=True)

    # train_set_for_1_labels = train_set_for_1_labels.drop(['prediction', 'confidence'], axis=1)
    train_set_for_1_labels.reset_index(drop=True, inplace=True)

    pargarphs0.reset_index(drop=True, inplace=True)

    new_data = pd.concat([predictions_of_1_high_conf, train_set_for_1_labels, pargarphs0], axis=0)
    from sklearn.utils import shuffle
    new_data = shuffle(new_data)
    return new_data

def predict_mails_from_paragraphs(paragraphs_with_prediction):
    mails_predictions = paragraphs_with_prediction.groupby(['document_id'])['prediction'].sum()
    mails_predictions[mails_predictions > 0] = 1
    mails_predictions.reset_index(drop=True, inplace=True)
    mails_ids = pd.DataFrame(np.unique(paragraphs_with_prediction[['document_id', 'label_id']], axis=0), columns=['document_id', 'label_id'])
    mails_ids.reset_index(drop=True, inplace=True)
    mails_results = pd.concat([mails_ids, mails_predictions], axis=1)
    return mails_results

def evaluate_results(mails_results, model_id):
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    model_file = 'results\ml_model_' + str(model_id) + '.pickle'
    model_path = data_path + '\\' + model_file
    clf = TextClassifier.load(model_path)
    X = None
    y = mails_results['label_id']
    y_pred = mails_results['prediction']
    _, evaluation_result_str = clf.evaluate(X, y, y_pred)
    return evaluation_result_str

def save_new_data(new_data, new_data_file):
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    new_data_path = os.path.join(data_path, new_data_file)
    with open(new_data_path, 'w') as f:
        new_data.to_csv(f, index=False)

def new_data_count(predictions_of_1_high_conf,  paragraphs0, train_set_for_1_labels):
    n1 = len(predictions_of_1_high_conf)
    n2 = len(paragraphs0)
    n3 = len(train_set_for_1_labels)
    print()
    print("New data info:")
    print("Number of new paragraphs predicted as 1 with high confidence: {}".format(n1))
    print("Number of new paragraphs from mails with label 0: {}".format(n2))
    print("Number of full mails with label 1: {}".format(n3))
    print("Total: {}".format(n1 + n2 + n3))


def bootstraping(filename, new_data_file, model_id):
    data = load_data(filename)
    paragraphs = textDf_2_tokens(data)
    paragraphs_with_prediction = predict_paragraphs(paragraphs, model_id)

    paragraphs1 = get_paragraphs_1_labels(paragraphs)
    # predictions_of_1_high_conf = get_pargarphs_predicted_1_with_high_conf(paragraphs_with_prediction, 0.5)
    predictions_of_1_high_conf = get_conf_labels(model_id, paragraphs1, 0.9, 0)
    train_set_for_1_labels = get_train_set_for_1_labels(data, predictions_of_1_high_conf)
    paragraphs0 = get_paragraphs_0_labels(paragraphs)
    new_data = create_new_dataset(predictions_of_1_high_conf, train_set_for_1_labels, paragraphs0)

    new_data_count(predictions_of_1_high_conf, paragraphs0, train_set_for_1_labels)

    save_new_data(new_data, new_data_file)

    mails_results = predict_mails_from_paragraphs(paragraphs_with_prediction)
    evaluation_result_str = evaluate_results(mails_results, model_id)


if __name__ == '__main__':
    filename = 'enron_ml_1_clean_shuffled_no_index2.csv'
    new_data_file = 'new_data_0016b.csv'

    model_id = '0016'

    bootstraping(filename, new_data_file, model_id)





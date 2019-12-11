import pandas as pd
import numpy as np
import os
from src.models.text_classifier import TextClassifier
from src.models.text_classifier import run_model_on_file
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import shuffle

def print_evaluation_scores(y, y_pred):
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print("# # # # # # # # # MODEL SCORES ON EXTERNAL TEST DATA # # # # # # # # # # #\n"
          " precision :{}   recall :{}    f1 :{}   on Test file".format(precision, recall, f1))

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

def get_paragraphs_predicted_1_with_high_conf(dataframe, confidence):
    # predictions_from_1_docs = dataframe.loc[dataframe['label_id'] == 1]
    predictions_of_1 = dataframe.loc[dataframe['prediction'] == 1]
    predictions_of_1_high_conf = predictions_of_1.loc[predictions_of_1['confidence'] > confidence]
    new_labels_df = predictions_of_1_high_conf[['document_id', 'label_id', 'text', 'user_id']]
    return new_labels_df

def get_train_set_for_1_labels(full_texts, paragarphs_predicted_1_with_high_conf):
    labeled_1_full_texts = full_texts[full_texts['label_id'] == 1]
    labeled_1_full_texts_not_in_par_pred = labeled_1_full_texts.loc[
        ~labeled_1_full_texts['document_id'].isin(paragarphs_predicted_1_with_high_conf['document_id'])]
    train_set_for_1_labels = pd.concat((labeled_1_full_texts_not_in_par_pred, paragarphs_predicted_1_with_high_conf),
                            axis=0, sort=True)
    return train_set_for_1_labels

def get_paragraphs_0_labels(paragraphs):
    paragraphs0 = paragraphs[paragraphs['label_id'] == 0]
    return paragraphs0

def get_texts_0_labels(texts):
    texts_0 = texts[texts['label_id'] == 0]
    return texts_0

def get_paragraphs_1_labels(paragraphs):
    paragraphs1 = paragraphs[paragraphs['label_id'] == 1]
    return paragraphs1

def create_new_dataset(train_set_for_1_labels, paragraphs0):
    # train_set_for_1_labels = train_set_for_1_labels.drop(['prediction', 'confidence'], axis=1)
    # train_set_for_1_labels.reset_index(drop=True, inplace=True)
    # paragraphs0.reset_index(drop=True, inplace=True)
    new_data = pd.concat([train_set_for_1_labels, paragraphs0], axis=0, sort=True)
    new_data = shuffle(new_data)
    new_data = new_data[['document_id', 'text', 'user_id', 'label_id']]
    return new_data

def save_new_data(new_data, new_data_file):
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    new_data_path = os.path.join(data_path, new_data_file)
    with open(new_data_path, 'w') as f:
        new_data.to_csv(f, index=False)

def predict_mails_from_paragraphs(paragraphs_with_prediction):
    paragraphs_with_prediction['prediction'] = paragraphs_with_prediction['prediction'].astype(int)
    mails_predictions = paragraphs_with_prediction.groupby(['document_id'])['prediction'].sum()
    mails_predictions[mails_predictions > 0] = 1
    mails_predictions.reset_index(drop=True, inplace=True)
    mails_ids = pd.DataFrame(np.unique(paragraphs_with_prediction[['document_id', 'label_id']], axis=0), columns=['document_id', 'label_id'])
    mails_ids.reset_index(drop=True, inplace=True)
    mails_ids.insert(0, 'prediction', mails_predictions)
    return mails_ids

def new_data_count(predictions_of_1_high_conf, paragraphs0, train_set_for_1_labels):
    n1 = len(predictions_of_1_high_conf)
    n2 = len(paragraphs0)
    n3 = len(train_set_for_1_labels)
    print()
    print("New data info:")
    print("Number of new paragraphs predicted as 1 with high confidence: {}".format(n1))
    print("Number of new paragraphs from mails with label 0: {}".format(n2))
    print("Number of full mails with label 1: {}".format(n3))
    print("Total: {}".format(n1 + n2 + n3))

def count_labels(data_frame_with_labels):
    label_count_1 = data_frame_with_labels.loc[data_frame_with_labels['label_id'] == 1]
    label_count_0 = data_frame_with_labels.loc[data_frame_with_labels['label_id'] == 0]
    print ("{} 1 labels in data. {} 0 labels in data".format(len(label_count_1), len(label_count_0)))

def bootstraping(filename, new_data_file, model_id, confidence_limit = 0.9, train_on_paragraphs = True):
    print('loading data set with full emails ', filename)
    data = load_data(filename)
    count_labels(data)
    print('parsing {} texts to paragraphs'.format(len(data)))
    paragraphs = textDf_2_tokens(data)
    count_labels(paragraphs)
    # predict only from 1 labeled texts
    paragraphs1 = get_paragraphs_1_labels(paragraphs)
    print('getting predictions for the {} paragraphs that came from texts originally labeled 1'.format(len(paragraphs1)))
    paragraphs_with_prediction = predict_paragraphs(paragraphs1, model_id)
    paragraphs1 = get_paragraphs_1_labels(paragraphs_with_prediction)
    predictions_of_1_high_conf = get_paragraphs_predicted_1_with_high_conf(paragraphs1, confidence_limit)
    print("number of paragraphs predicted 1 with {} confidence is {}".format(confidence_limit,
                                                                             len(predictions_of_1_high_conf)))
    # predictions_of_1_high_conf = get_conf_labels(model_id, paragraphs1, confidence_limit, 0)
    train_set_for_1_labels = get_train_set_for_1_labels(data, predictions_of_1_high_conf)
    print("number of texts in bootstraped dataset labeled 1 is {} ".format(len(train_set_for_1_labels)))
    if train_on_paragraphs == True:
        train_set_for_0_labels = get_paragraphs_0_labels(paragraphs)
    if train_on_paragraphs == False:
        train_set_for_0_labels = get_texts_0_labels(data)
    print("number of texts in bootstraped dataset labeled 0 is {} ".format(len(train_set_for_0_labels)))
    new_data = create_new_dataset(train_set_for_1_labels, train_set_for_0_labels)
    new_data_count(predictions_of_1_high_conf, train_set_for_0_labels, train_set_for_1_labels)
    save_new_data(new_data, new_data_file)

def run_model_on_texts(file_path, project_id):
    print ("run model on full texts to create classifier")
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    input_file = os.path.join(data_path, file_path)
    output_file = os.path.join(data_path, 'output.csv')
    result = run_model_on_file(
        input_filename=input_file,
        output_filename=output_file,
        project_id=project_id,
        user_id=2,
        label_id=None,
        run_on_entire_dataset=False)

def load_model_from_file(model_id):
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    model_file = 'results\ml_model_' + str(model_id) + '.pickle'
    model_path = data_path + '\\' + model_file
    clf = TextClassifier.load(model_path)
    return clf

if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    #path for file with full text to train model on
    full_texts_filename = '\enron_with_categories\enron_cat11_single_clean.csv'
    model_id = '0010'
    new_data_file = 'new_data_0010b.csv'
    #choose confidence limit to give pargraphs label 1
    confidence_limit = 0.9
    train_on_paragraphs = True # True: 0 labeled text are paragraphs
    # create train and test files
    full_texts_df = pd.read_csv(data_path + full_texts_filename)
    train_df = full_texts_df.sample(frac=0.9, random_state=0)
    test_df = full_texts_df.loc[~full_texts_df['document_id'].isin(train_df['document_id'])]
    # save train_df
    export_csv = train_df.to_csv(data_path + r"\train_set", index=False, header=True)
    # train - model on full texts
    run_model_on_texts(data_path + r"\train_set", model_id)
    # get scores for test file with original model
    model_clf = load_model_from_file(model_id)
    pre_processed = model_clf.pre_process(test_df, fit=False)
    predictions = model_clf.predict(pre_processed).astype(int)
    print_evaluation_scores(test_df['label_id'], predictions)
    # bootstrap data
    bootstraping(data_path + r"\train_set", new_data_file, model_id, confidence_limit,train_on_paragraphs)
    # train model on bootsrap data
    run_model_on_texts(new_data_file, str(int(model_id) + 1))
    print("model saved with id {}".format(str(int(model_id) + 1)))
    ##################### validation on test file ####################
    # get scores for test file
    print ("scores on data of full texts")
    model_clf = load_model_from_file(str(int(model_id) + 1))
    pre_processed = model_clf.pre_process(test_df, fit=False)
    predictions = model_clf.predict(pre_processed).astype(int)
    predictions_df = model_clf.get_prediction_df(pre_processed)
    print_evaluation_scores(test_df['label_id'], predictions)
    print("scores on data of paragraphed texts")
    paragraphed_test_text_df = textDf_2_tokens(test_df)
    paragraphed_pre_processed = model_clf.pre_process(paragraphed_test_text_df, fit=False)
    paragraphed_predictions = model_clf.predict(paragraphed_pre_processed).astype(int)
    paragraphed_predictions_df = model_clf.get_prediction_df(paragraphed_pre_processed)
    paragraphed_predictions_df.insert(0, 'document_id', paragraphed_test_text_df['document_id'])
    paragraphed_predictions_df.insert(0, 'label_id', paragraphed_test_text_df['label_id'])
    print("scores on pargraphs from test data")
    print_evaluation_scores(paragraphed_predictions_df['label_id'], paragraphed_predictions)
    predicted_by_mail = predict_mails_from_paragraphs(paragraphed_predictions_df)
    print("scores on mails predicted from paragraphs  on test data")
    print_evaluation_scores(test_df['label_id'], predicted_by_mail['prediction'])





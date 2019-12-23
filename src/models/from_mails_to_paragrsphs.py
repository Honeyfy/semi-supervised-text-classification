'''
Code for transfering labels from full mails to paragraphs labels.
A new dataset is generated.
Inputs: trained model id, data file name, and new data file name for saving the new data
All paragraphs from mails with labels 0 automatically labeled as 0.
All paragraphs predicted as 1 with confidence above threshold from the given trained model labeled as 1.
All full mails, with lables 1, which did not have paragraphs which were labeled with high confidence,
are added to the new dataset as well.

This project is collaboration of Roi Ruach, Ran Dan, Amir Hamenahem and Shira Weissman.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.bootstrap_classifier import run_model_on_train_test_split


def split_text_to_paragraphs(data, pattern='\n\n'):
    # Splitting text to paragraphs separated with two lines in the full text
    data['text'] = data['text'].str.split(pattern, expand=False)
    paragraphs = data.explode('text').reset_index(drop=True)
    paragraphs.dropna(subset=['text'], inplace=True)
    return paragraphs

def predict_paragraphs(paragraphs, bootstrap_clf):
    X = bootstrap_clf.pre_process(paragraphs, fit=False)
    paragraphs = paragraphs.drop(['processed_text'], axis=1)
    paragraphs_pred = bootstrap_clf.get_prediction_df(X)
    paragraphs_pred.reset_index(drop=True, inplace=True)
    paragraphs.reset_index(drop=True, inplace=True)
    paragraphs_with_prediction = pd.concat([paragraphs, paragraphs_pred], axis=1)
    paragraphs_with_prediction['prediction'] = paragraphs_with_prediction['prediction'].astype(int)
    return paragraphs_with_prediction

def predict_mails_from_paragraphs(paragraphs_with_prediction):
    mails_predictions = paragraphs_with_prediction.groupby(['document_id'])['prediction'].sum()
    mails_predictions[mails_predictions > 0] = 1
    mails_predictions.reset_index(drop=True, inplace=True)
    paragraphs_with_prediction['label_id'] = paragraphs_with_prediction['label_id'].astype(int)
    mails_ids = pd.DataFrame(np.unique(paragraphs_with_prediction[['document_id', 'label_id']], axis=0), columns=['document_id', 'label_id'])
    mails_ids.reset_index(drop=True, inplace=True)
    mails_results = pd.concat([mails_ids, mails_predictions], axis=1)
    return mails_results

def evaluate_results(mails_results, bootstrap_clf):
    print("Evaluation report for classifying full mails from paragraphs predictions:")
    X = None
    y = mails_results['label_id']
    y_pred = mails_results['prediction']
    _, evaluation_result_str = bootstrap_clf.evaluate(X, y, y_pred)
    return evaluation_result_str


def paragraphs_data_count(predictions_of_positive_high_conf,  paragraphs_negative, train_set_for_positive_labels):
    n1 = len(predictions_of_positive_high_conf)
    n2 = len(paragraphs_negative)
    n3 = len(train_set_for_positive_labels)
    print()
    print("New data info:")
    print("Number of new paragraphs predicted as positive with high confidence: {}".format(n1))
    print("Number of new paragraphs from mails with negative label: {}".format(n2))
    print("Number of full mails with positive label: {}".format(n3))
    print("Total: {}".format(n1 + n2 + n3))


def generate_paragraphs_dataset(data, paragraphs_with_prediction, positive_confidence=0.9):
    high_conf_pos_predictions = (paragraphs_with_prediction['label_id'] == 1) & \
                                (paragraphs_with_prediction['prediction'] == 1) & \
                                (paragraphs_with_prediction['confidence'] > positive_confidence)
    predictions_of_positive_high_conf = paragraphs_with_prediction.loc[high_conf_pos_predictions]
    predictions_of_positive_high_conf = predictions_of_positive_high_conf[['document_id', 'text', 'user_id', 'label_id']]

    train_set_for_positive_labels = data.loc[(paragraphs_with_prediction['label_id'] == 1) &
                                             ~data['document_id'].isin(predictions_of_positive_high_conf['document_id'])]

    paragraphs_negative = paragraphs_with_prediction[(paragraphs_with_prediction['label_id'] == 0) &
                                                     (paragraphs_with_prediction['text'] != '') &
                                                     (paragraphs_with_prediction['text'] != ' ')]
    paragraphs_negative = paragraphs_negative[['document_id', 'text', 'user_id', 'label_id']]
    paragraphs_data = pd.concat([predictions_of_positive_high_conf, train_set_for_positive_labels, paragraphs_negative],
                                axis=0, sort=False)
    paragraphs_data = paragraphs_data.sample(frac=1).reset_index()
    paragraphs_data_count(predictions_of_positive_high_conf, paragraphs_negative, train_set_for_positive_labels)
    paragraphs_data['text'] = paragraphs_data['text'].astype(str)
    return paragraphs_data


def from_mails_to_paragraphs_all_included(full_mails_data_file_path, paragraphs_data_file_path,
                                          predictions_of_paragraphs_from_full_mails_file_path,
                                          full_mails_model_output_file,
                                          paragraphs_model_output_file,
                                          full_mails_model_id,
                                          full_mails_C_parameter,
                                          paragraphs_positive_prediction_confidence_threshold,
                                          paragraphs_model_id,
                                          paragraphs_C_parameter):

    ### Load data and prepare it for training:
    ##########################################
    with open(full_mails_data_file_path, 'r') as f:
        full_email_data = pd.read_csv(f)

    label_column_name = "label_id"
    data_column_names = list(full_email_data.columns.values)
    data_column_names.remove(label_column_name)

    label_field = 'label_id'
    if 'label_id' in full_email_data.columns:
        full_email_data['label'] = full_email_data['label_id']
    elif 'label' not in full_email_data.columns:
        raise ValueError("no columns 'label' or 'label_id' exist in input file")

    df = full_email_data[~pd.isnull(full_email_data['text'])]

    df.loc[:, label_field] = df[label_field].apply(lambda x: str(x) if not pd.isnull(x) else x)
    df.loc[df[label_field] == ' ', label_field] = None

    df_labeled = df[(~pd.isnull(df[label_field]))]
    df_labeled = df_labeled[data_column_names + [label_column_name]]

    X_train, X_test, y_train, y_test = train_test_split(df_labeled[data_column_names], df_labeled[label_column_name],
                                                        test_size=0.25)

    y_train = y_train.astype(str)
    y_test = y_test.astype(str)

    print()
    print("""Training the full mails data in BootstrapClassifier:""")
    print("""########################################################""")
    print()
    result, bootstrap_clf = run_model_on_train_test_split(
        df_labeled, X_train, X_test, y_train, y_test,
        output_filename=full_mails_model_output_file,
        user_id=2,
        project_id=full_mails_model_id,
        C=full_mails_C_parameter,
        label_id=None,
        method='bow',
        run_on_entire_dataset=False)

    print()
    print("""Evaluation of the full mails predictions from paragraphs labels.""")
    print("""Paragraphs labels predicted by the full mails classifier.""")
    print("""###################################################################""")
    print()

    train_data = df_labeled.loc[X_train.index]
    test_data = df_labeled.loc[X_test.index]

    paragraphs_train = split_text_to_paragraphs(train_data)
    paragraphs_test = split_text_to_paragraphs(test_data)

    # using model trained on complete emails to predict on paragraphs
    paragraphs_with_prediction_train = predict_paragraphs(paragraphs_train, bootstrap_clf)
    paragraphs_with_prediction_test = predict_paragraphs(paragraphs_test, bootstrap_clf)

    mails_results_train = predict_mails_from_paragraphs(paragraphs_with_prediction_train)
    mails_results_test = predict_mails_from_paragraphs(paragraphs_with_prediction_test)
    print('Evaluation report for train data with the classifier trained on full mails:')
    _ = evaluate_results(mails_results_train, bootstrap_clf)
    print()
    print('Evaluation report for test data with the classifier trained on full mails:')
    _ = evaluate_results(mails_results_test, bootstrap_clf)


    paragraphs_with_prediction_total = pd.concat([paragraphs_with_prediction_train, paragraphs_with_prediction_test], axis=0)

    with open(predictions_of_paragraphs_from_full_mails_file_path, 'w') as f:
        paragraphs_with_prediction_total.to_csv(f, index=False)

    ### Generating the paragraphs dataset which includes:
    ### 1. Paragraphs with negative labels from mails with negative labels.
    ### 2. Paragraphs with positive labels predicted with high confidence from the full mails classifier.
    ### 3. Full mails with positive labels which none of their paragraphs was predicted as positive wiht high confidence.
    #####################################################################################################################
    print()
    print("""Generating the paragraphs dataset..""")
    print("""##################################""")
    print()

    positive_confidence = paragraphs_positive_prediction_confidence_threshold

    paragraph_train_with_labels_df = generate_paragraphs_dataset(train_data, paragraphs_with_prediction_train,
                                                                  positive_confidence)
    paragraph_test_with_labels_df = generate_paragraphs_dataset(test_data, paragraphs_with_prediction_test,
                                                                positive_confidence)

    X_train_paragraphs = paragraph_train_with_labels_df[data_column_names]
    y_train_paragraphs = paragraph_train_with_labels_df[label_column_name]

    X_test_paragraphs = paragraph_test_with_labels_df[data_column_names]
    y_test_paragraphs = paragraph_test_with_labels_df[label_column_name]

    y_train_paragraphs = y_train_paragraphs.astype(str)
    y_test_paragraphs = y_test_paragraphs.astype(str)

    paragraphs_dataset = pd.concat([paragraph_train_with_labels_df, paragraph_test_with_labels_df], axis=0)

    with open(paragraphs_data_file_path, 'w') as f:
        paragraphs_dataset.to_csv(f, index=False)

    print()
    print("""Training the paragraphs dataset in BootstrapClassifier:""")
    print("""#######################################################""")
    print()

    paragraphs_result, paragraphs_bootstrap_clf = run_model_on_train_test_split(df_labeled,
        X_train_paragraphs, X_test_paragraphs, y_train_paragraphs, y_test_paragraphs,
        output_filename=paragraphs_model_output_file,
        user_id=2,
        project_id=paragraphs_model_id,
        C=paragraphs_C_parameter,
        label_id=None,
        method='bow',
        run_on_entire_dataset=False)

    print()
    print("""Evaluation of the full mails predictions from paragraphs labels.""")
    print("""Paragraphs labels predicted by the paragraphs classifier.""")
    print("""####################################################################""")
    print()

    paragraphs_with_prediction_train_paragraphs_classifier = predict_paragraphs(paragraphs_train,
                                                                                paragraphs_bootstrap_clf)
    paragraphs_with_prediction_test_paragraphs_classifier = predict_paragraphs(paragraphs_test,
                                                                               paragraphs_bootstrap_clf)

    mails_results_train_paragraphs_classifier = predict_mails_from_paragraphs(
        paragraphs_with_prediction_train_paragraphs_classifier)

    mails_results_test_paragraphs_classifier = predict_mails_from_paragraphs(
        paragraphs_with_prediction_test_paragraphs_classifier)

    print('Evaluation report for train data with the classifier trained on paragraphs:')
    _ = evaluate_results(mails_results_train_paragraphs_classifier, paragraphs_bootstrap_clf)
    print()
    print('Evaluation report for test data with the classifier trained on paragraphs::')
    _ = evaluate_results(mails_results_test_paragraphs_classifier, paragraphs_bootstrap_clf)


if __name__ == '__main__':
    full_mails_data_file_path = \
        r'C:\develop\code\semi-supervised-text-classification\data\enron_ml_1_clean_shuffled_no_index2.csv'
    paragraphs_data_file_path = r'C:\develop\code\semi-supervised-text-classification\data\paragraphs_dataset.csv'
    predictions_of_paragraphs_from_full_mails_file_path = \
        r'C:\develop\code\semi-supervised-text-classification\data\results\predictions_of_paragraphs_from_full_mails.csv'
    full_mails_model_output_file = r'C:\develop\code\semi-supervised-text-classification\data\bootstrap_full_mails_output.csv'
    paragraphs_model_output_file = r'C:\develop\code\semi-supervised-text-classification\data\bootstrap_paragraphs_output.csv'


    full_mails_model_id = 82
    full_mails_C_parameter = 2

    paragraphs_positive_prediction_confidence_threshold = 0.8


    paragraphs_model_id = 83
    paragraphs_C_parameter = 1


    from_mails_to_paragraphs_all_included(full_mails_data_file_path, paragraphs_data_file_path,
                                          predictions_of_paragraphs_from_full_mails_file_path,
                                          full_mails_model_output_file,
                                          paragraphs_model_output_file,
                                          full_mails_model_id,
                                          full_mails_C_parameter,
                                          paragraphs_positive_prediction_confidence_threshold,
                                          paragraphs_model_id,
                                          paragraphs_C_parameter)



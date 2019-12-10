
import pandas as pd
from src.models.text_classifier import TextClassifier
import numpy as np

def get_conf_labels(trained_model,par_text_df, min_confidence, user_id, include_zeros=True):
    one_string = ''       #TODO what hapens when b<1?
    clf = trained_model
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







if __name__ == '__main__':
    i=0
    min_confidence = 0.85
    user_id = 1
    num = str(i)
    name = str('new_labels_')+num
    sample_size = 500
    model_path = r'C:\develop\code\semi-supervised-text-classification\data\results\ml_model_'
    all_par_data = pd.read_csv(
        r'C:\develop\code\semi-supervised-text-classification\data\expand_samp_enron_no_head.csv')

    trained_model_path = model_path+name+'.pickle'
    trained_model = TextClassifier.load(trained_model_path)
    par_sample = all_par_data.sample(sample_size)

    new_labels_df, conf = get_conf_labels(trained_model, par_sample, min_confidence, user_id, include_zeros=True)

    print('a')
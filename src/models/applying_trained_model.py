import pandas as pd
from src.models.text_classifier import TextClassifier

def apply_trained_model(model_id, data_filename):
    # a function applying a trained model to a new csv file and returning a dataframe with predicted label and confidence.
    model_path = r'C:\develop\code\semi-supervised-text-classification\data\results\ml_model_' + str(model_id) + '.pickle'

    clf = TextClassifier.load(model_path)

    file_path = r'C:\develop\code\semi-supervised-text-classification\data' + '\\' + data_filename
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
    X = clf.pre_process(df, fit=False)
    df_pred = clf.get_prediction_df(X)
    df_with_prediction = pd.concat([df, df_pred], axis=1)
    return df_with_prediction

if __name__ == '__main__':
    model_id = '1111'
    data_filename = 'enron_ml_1.csv'
    df_with_prediction = apply_trained_model(model_id, data_filename)








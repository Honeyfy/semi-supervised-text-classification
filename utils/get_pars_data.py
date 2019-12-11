import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


from src.models.text_classifier import run_model_on_file
from src.models.text_classifier import TextClassifier
from utils import get_conf_labels as get_conf_labels
from utils.get_mails import get_rand_mail
from utils.clean_n_split import clean_n_split

# data_path = r'C:\develop\code\semi-supervised-text-classification\data'
# model_path = r'C:\develop\code\semi-supervised-text-classification\data\results\ml_model_'
# full_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\enron_no_head.csv')
# all_par_data = clean_n_split(full_data)

# # Hyper parm
# iters = 50
# sample_size = 2000
# min_confidence = 0.85
# conf_ls = []
# result_ls = []



def get_par_data(data_path,model_path,true_data,name,min_confidence,mails_num,user_id):



    input_file = os.path.join(data_path, 'enron_no_head.csv')
    output_file = os.path.join(data_path, 'output.csv')
    training_df = pd.read_csv(input_file)


    result = run_model_on_file(
        input_filename=input_file,
        output_filename=output_file,
        project_id=name,
        user_id=2,
        label_id=None,
        run_on_entire_dataset=False)

    trained_model_path = model_path + name + '.pickle'
    trained_model = TextClassifier.load(trained_model_path)



    true_par_sample, mail_id = get_rand_mail(true_data, mails_num)
    mail_id=np.unique(mail_id)
    #df = df[df['Column Name'].isin(['Value']) == False]
    true_data = true_data.loc[true_data['document_id'].isin(mail_id) == False]

    new_labeld_df = get_conf_labels.get_conf_labels(trained_model, true_par_sample, min_confidence, user_id,include_zeros=True)
    return  new_labeld_df




if __name__ == '__main__':
    data_path = r'C:\develop\code\semi-supervised-text-classification\data'
    model_path = r'C:\develop\code\semi-supervised-text-classification\data\results\ml_model_'
    full_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\enron_no_head.csv')
    all_par_data = clean_n_split(full_data)

    user_id = 0
    mails_num = 20
    min_confidence = 0.85
    name = '0'
    new_labeld_df = get_par_data(all_par_data, name, min_confidence, mails_num, user_id)



    print('a')
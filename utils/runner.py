import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src.models.text_classifier import run_model_on_file
from src.models.text_classifier import TextClassifier
from utils import get_conf_labels as get_conf_labels
from utils.bootstrap import my_bootstrap
from utils.get_pars_data import get_par_data


from utils.clean_n_split import clean_n_split


def par_runner(all_par_data,data_path,model_path,mails_num,min_confidence,model_name,user_id):
    name = model_name
    i = -1
    true_data = all_par_data.loc[all_par_data['label_id'] == 1.]
    false_data = all_par_data.loc[all_par_data['label_id'] == 0.]

    while len(true_data) > 0:
        i += 1
        mails_remain = len(np.unique(true_data['document_id']))
        if mails_remain <= mails_num:
            mails_num = mails_remain

        par_data = get_par_data(data_path, model_path, true_data, name, min_confidence, mails_num, user_id)

        if i == 0:
            full_par_data = par_data
        else:
            full_par_data = pd.concat([full_par_data, par_data], ignore_index=True)

    full_par_data = pd.concat([full_par_data, false_data], ignore_index=True)
    return full_par_data



if __name__ == '__main__':
    data_path = r'C:\develop\code\semi-supervised-text-classification\data'
    model_path = r'C:\develop\code\semi-supervised-text-classification\data\results\ml_model_'
    full_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\enron_no_head.csv')
    all_par_data = clean_n_split(full_data)

    # Hyper parm
    mails_num = 20
    min_confidence = 0.85
    model_name = '0'
    user_id = 2


    full_par_data = par_runner(all_par_data, data_path, model_path, mails_num, min_confidence, model_name, user_id)

    true_data = all_par_data.loc[all_par_data['label_id'] == 1.]
    false_data = all_par_data.loc[all_par_data['label_id'] == 0.]

    while len(true_data) > 0:
        i +=1
        mails_remain = len(np.unique(true_data['document_id']))
        if mails_remain<=mails_num:
            mails_num = mails_remain

        par_data = get_par_data(data_path,model_path,true_data,name,min_confidence,mails_num,user_id)

        if i==0:
            full_par_data = par_data
        else:
            full_par_data = pd.concat([full_par_data,par_data],ignore_index=True)

    full_par_data = pd.concat([full_par_data,false_data],ignore_index=True)


    print('a')
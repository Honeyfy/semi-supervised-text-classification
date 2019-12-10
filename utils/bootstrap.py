import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from src.models.text_classifier import run_model_on_file
from src.models.text_classifier import TextClassifier
from utils import get_conf_labels as get_conf_labels
from utils.get_mails import get_rand_mail
# data_path = r'C:\develop\code\semi-supervised-text-classification\data'
# model_path = r'C:\develop\code\semi-supervised-text-classification\data\results\ml_model_'
# all_par_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\expand_samp_enron_no_head.csv')


# # Hyper parm
# iters = 50
# sample_size = 2000
# min_confidence = 0.85
# conf_ls = []
# result_ls = []

def my_bootstrap(data_path,model_path,all_par_data,iters,sample_size,min_confidence):
    pred_ls = []
    result_ls = []

    true_data = all_par_data.loc[all_par_data['label_id'] == 1.]
    false_data = all_par_data.loc[all_par_data['label_id'] == 0.]


    for i in range(iters):
        print('###################################  ',i, '  ##################')


        if i ==0:
            input_file = os.path.join(data_path, 'enron_no_head.csv')
            training_df = pd.read_csv(input_file)


        else:
            num = str(i)
            name = str('enron_new_labels_')+num+'.csv'
            input_file = os.path.join(data_path, name)
            training_df = pd.read_csv(input_file)

        output_file = os.path.join(data_path, 'output.csv')
        num = str(i)
        name = str('new_labels_')+num

        result = run_model_on_file(
            input_filename=input_file,
            output_filename=output_file,
            project_id=name ,
            user_id=2,
            label_id=None,
            run_on_entire_dataset=False)

        result_ls.append(result.split('Performance on test set:')[1])

        trained_model_path = model_path+name+'.pickle'
        trained_model = TextClassifier.load(trained_model_path)

        mails_num = 3
        par_sample,mail_id = get_rand_mail(all_par_data,mails_num)
        true_lable = par_sample.iloc[0,1]
        new_labeld_df, pred,one_string = get_conf_labels.get_conf_labels(trained_model, par_sample, min_confidence, i, include_zeros=True)
        pred_ls.append([mail_id,one_string,pred,pred==true_lable])


        if len(new_labeld_df) >0:
            training_df = training_df.loc[training_df['document_id'] != mail_id].reset_index(drop=True)
            new_data = pd.concat([training_df,new_labeld_df],ignore_index=True)

        num = str(i+1)
        name = str('enron_new_labels_')+num+'.csv'
        output_path = os.path.join(data_path, name)
        new_data.to_csv(output_path,index=False)


        # print(conf_ls[0],conf_ls[-1])
        # print(result_ls[0])
        # print(result_ls[-1])



    return pred_ls,result_ls,trained_model_path

if __name__ == '__main__':
     data_path = r'C:\develop\code\semi-supervised-text-classification\data'
     model_path = r'C:\develop\code\semi-supervised-text-classification\data\results\ml_model_'
     all_par_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\expand_samp_enron_no_head.csv')
     # Hyper parm
     iters = 2
     sample_size = 2000
     min_confidence = 0.85
     conf_ls = []
     result_ls = []

     conf_ls,result_ls,trained_model_path = my_bootstrap(data_path,model_path,all_par_data,iters,sample_size,min_confidence)

     print(result_ls[0])
     print(result_ls[-1])


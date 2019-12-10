import numpy as np
import pandas as pd
from utils.clean_n_split import clean_n_split

def get_rand_mail(all_par_data,mails_num):
    documents = np.unique(all_par_data['document_id'])
    indexes = np.random.randint(0,len(documents),mails_num)
    rand_mail = all_par_data.loc[all_par_data['document_id'].isin(documents[indexes] )]
    mails_id = rand_mail['document_id']
    return rand_mail,mails_id



if __name__ == '__main__':
    full_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\enron_ml_1_clean.csv')
    all_par_data = clean_n_split(full_data,cut_header=False)

    rand_maills = get_rand_mails(all_par_data,1)

    print('a')

    mail_id = rand_maills['document_id'][0]
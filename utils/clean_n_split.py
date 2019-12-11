import pandas as pd


def clean_n_split(data,cut_header=False):
    if cut_header==True:
        data['text'] = data['text'].str.split(r'X-FileName:', expand=True).iloc[:, 1]

    # split to paragraphs
    data['text'] = data['text'].str.split('\n\r')
    data =data.explode('text').reset_index(drop=True)
    data = data.loc[data['text'] != "", :]
    return data




if __name__== '__main__':
    data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\enron_ml_1_clean.csv')
    clean_data = clean_n_split(data)

    print('a')
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import classification_report


"""
job: vanilla binary CE_loss
input: pd.DataFrame,torch.tensor
output:torch.tensor
"""
def clean_consistency_loss(data, a):
    data = data.drop('origin', axis=1)
    y_true = data['target']
    x = data.drop(['target', 'document_id'], axis=1)
    y_true = torch.tensor(y_true.values)
    x = torch.tensor(x.values)
    # print(origin_x.shape,a.shape)
    z = x @ a.double()
    y_hat = torch.sigmoid(z)
    CE_loss = F.binary_cross_entropy(y_hat, y_true)
    return CE_loss


"""
job: consistency loss 
input: pd.DataFrame,torch.tensor
output:torch.tensor
"""
def consistency_loss(data,a):
  #print('consistency_loss')
  origin_data = data.loc[data['origin']==True,:]
  aug_data = data.loc[data['origin']==False,:]

  origin_data = origin_data.drop(['origin','document_id'],axis=1)
  aug_data = aug_data.drop('origin',axis=1)

  aug_1,aug_2 = split_df(aug_data)
  aug_1.drop('document_id',axis=1,inplace=True)
  aug_2.drop('document_id',axis=1,inplace=True)

  origin_y = origin_data['target']
  origin_x = origin_data.drop('target',axis=1)
  aug_1_x = aug_1.drop('target',axis=1)
  # print(aug_1_x.columns)
  aug_2_x = aug_2.drop('target',axis=1)

  origin_y = torch.tensor(origin_y.values)
  origin_x = torch.tensor(origin_x.values)
  aug_1_x = torch.tensor(aug_1_x.values)
  aug_2_x = torch.tensor(aug_2_x.values)

  # print(origin_x.shape,a.shape)
  z= origin_x@a.double()
  y_hat = torch.sigmoid(z)
  #todo - ____ here


  origin_CE_loss = F.binary_cross_entropy(y_hat,origin_y)
  z_1= aug_1_x@a.double()
  y_1_hat = torch.sigmoid(z).detach()

  z_2= aug_2_x@a.double()
  y_2_hat = torch.sigmoid(z).detach()
  # inputs = torch.double(y_1_hat.numpy().double(),y_2_hat.numpy().double())
  #aug_CE_loss = F.cross_entropy(inputs,target)
  #aug_CE_loss = F.binary_cross_entropy(y_1_hat,y_2_hat)
  #aug_CE_loss = F.binary_cross_entropy(y_hat.,conf)

  loss = nn.CrossEntropyLoss()
  y_1_hat = torch.unsqueeze(y_1_hat,-1)
  y_2_hat = torch.unsqueeze(y_2_hat,-1)
  inputs = torch.cat((y_1_hat, y_2_hat),1).double()
  targets = y_hat.long()
  output = loss(inputs, targets)

  aug_CE_loss = output
  return origin_CE_loss +  0*aug_CE_loss


"""
job: split dataFrame to original and augmented dataFrames
input: pd.DataFrame
output: pd.DataFrame,pd.DataFrame
"""
def split_df(aug_data):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    df_ls = [df1, df2]  # TODO replace loops
    for doc in set(aug_data['document_id']):
        df = aug_data.loc[aug_data['document_id'] == doc, :]
        for j in range(len(df)):
            row = df.iloc[j, :]
            df_ls[j] = df_ls[j].append(df.iloc[j, :])

    return df_ls[0], df_ls[1]


"""
job: calc loss and update weights 
input: 
output:
"""
def update():
    loss = clean_consistency_loss(processed_train_data,weights)
    #loss = consistency_loss(train_data,a)
    if t % 100 == 0: print(loss)
    loss.backward()
    with torch.no_grad():
        # a.add_(lr * a.grad) #add
        weights.sub_(lr * weights.grad) #sub
        weights.grad.zero_()

"""
job: returns  train_data, processed_train_data, test_data, processed_test_data
input: 
output:pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame
"""
def get_data():
    train_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\origin_train_hotel_reviews.csv')
    test_data = pd.read_csv(r'C:\develop\code\semi-supervised-text-classification\data\origin_test_hotel_reviews.csv')
    print('train', train_data.shape, 'test', test_data.shape)
    train_data.rename(columns={'label_id': 'target'}, inplace=True)
    test_data.rename(columns={'label_id': 'target'}, inplace=True)
    train_data['target'].mean()

    train_data['origin']=True
    test_data['origin'] = True


    processed_train_data = pd.read_csv(r'C:/develop/code/semi-supervised-text-classification/data/processed_origin_train_hotel_reviews.csv')
    processed_test_data = pd.read_csv(r'C:/develop/code/semi-supervised-text-classification/data/processed_origin_test_hotel_reviews.csv')

    processed_train_data['origin'] = train_data['origin'].copy()
    processed_train_data['document_id'] = train_data['document_id'].copy()
    processed_train_data.loc[processed_train_data['target'] == 7, 'target'] = 1
    processed_train_data.loc[processed_train_data['target'] == 6, 'target'] = 0
    processed_train_data.head(5)

    processed_test_data['origin'] = test_data['origin'].copy()
    processed_test_data['document_id'] = test_data['document_id'].copy()
    processed_test_data.loc[processed_test_data['target'] == 7, 'target'] = 1
    processed_test_data.loc[processed_test_data['target'] == 6, 'target'] = 0
    processed_test_data.head(5)

    return train_data, processed_train_data,test_data,processed_test_data

"""
job: gets processed_test_data, weights, threshold returns prediction
input: pd.DataFrame,torch.Tensor,float(0,1)
output:pd.DataFrame
"""
def predict(processed_test_data,weights,threshold):
    resoults = pd.DataFrame(processed_test_data['target'].copy())
    x1 = torch.tensor(processed_test_data.drop(['target', 'document_id', 'origin'], axis=1).values)
    z = x1 @ weights.double()
    y_hat = torch.sigmoid(z).detach().numpy()
    resoults['y_hat'] = y_hat
    resoults['pred'] = y_hat > threshold
    resoults['calc'] = resoults['pred'] == resoults['target']
    y_t = resoults['target'].astype('int').values
    y_pred = resoults['pred'].values
    target_names = ['class 0', 'class 1']
    print(classification_report(y_t, y_pred, target_names=target_names))
    return resoults





if __name__ == '__main__':

    train_data, processed_train_data, test_data, processed_test_data = get_data()

    # init weights
    weights = torch.ones(5000)
    weights.requires_grad = True

    lr = 10

    # lr = 1
    print('\nperforming gradient descent')
    for t in range(500):
        update()

    resoults = predict(processed_test_data, weights, threshold=0.5)
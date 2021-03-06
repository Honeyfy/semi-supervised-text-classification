'''
A quick edit to text_classifier.py.
bootstrap_classifier is used for the paragraph bootstrap project.
added the function: run_model_on_train_test_split
input:
X_train, X_test, y_train, y_test, output_filename, user_id, project_id,
label_id=None, method='bow', run_on_entire_dataset=False)

return:
result, clf
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.models.base_classifier import BaseClassifier
from src.models.text_pipeline import TextPipeline
from sklearn.linear_model import LogisticRegression
import logging
import matplotlib.pyplot as plt

from src.utils.analyze_model import plot_precision_recall_curve, plot_roc_curve, plot_confidence_performance

logger = logging.getLogger('text classifier')
ngram_range = (1, 2)
# ngram_range = (1, 2)

RUN_FASTTEXT = False


class BootstrapClassifier(BaseClassifier):
    @classmethod
    def load(cls, filename):
        # super loads only the model
        classifier, offset = super().load(filename)

        # inherited class loads the pipeline
        try:
            processing_pipeline = TextPipeline.load(filename, offset)
        except EOFError:
            logger.warning('EOF reached when trying to load pipeline')
            processing_pipeline = None
        classifier.processing_pipeline = processing_pipeline
        return classifier

    @property
    def important_features(self, k=None):
        column_order = ['feature_name', 'importance', 'class']
        try:
            importances = self._model.feature_importances_
            feature_names = self.features
            feature_importance_df = pd.DataFrame({'feature_name': feature_names,
                                                  'importance': importances,
                                                  'class': np.nan}).sort_values(by='importance', ascending=False)
            if isinstance(k, int):
                feature_importance_df = feature_importance_df.head(k)
        except:
            feature_names = self.features
            if len(self._model.classes_) > 2:
                feature_importances_per_class = []
                for i, c in enumerate(self._model.classes_):
                    importances = self._model.coef_[i]
                    per_class_df = pd.DataFrame({'feature_name': feature_names[importances > 0],
                                                 'importance': importances[importances > 0],
                                                 'class': c}).sort_values(by='importance', ascending=False)
                    if isinstance(k, int):
                        per_class_df = per_class_df.head(k)
                    feature_importances_per_class.append(per_class_df)

                feature_importance_df = pd.concat(feature_importances_per_class, sort=True)
            else:
                importances = self._model.coef_[0]
                feature_importance_df = pd.DataFrame({'feature_name': feature_names,
                                                      'importance': abs(importances),
                                                      'class': [
                                                          self._model.classes_[0] if imp < 0 else self._model.classes_[
                                                              1] for imp in importances]})

        return feature_importance_df.sort_values(by=['class', 'importance'], ascending=False)[column_order]

    def set_preprocessor(self, pipeline):
        self.processing_pipeline = TextPipeline(pipeline)

    def run_on_train_test(self, df, X_train_in, X_test_in, y_train_in, y_test_in, output_filename, user_id, project_id,
                          label_id=None, pipeline=None, bootstrap_iterations=0, bootstrap_threshold=0.9,
                          run_on_entire_dataset=False):

        output_filename = os.path.abspath(output_filename)
        output_folder = os.path.join(os.path.dirname(output_filename), 'results')
        os.makedirs(output_folder, exist_ok=True)

        print('Running text classification model on train test split. Results will be saved to {}...'.format(output_filename))
        print('Reading input file...')

        label_field = 'label_id'
        self.set_preprocessor(pipeline)

        # # concat train test split to single dataframe
        # X_train_in.insert(X_train_in.shape[1], label_field, y_train_in)
        # X_test_in.insert(X_test_in.shape[1], label_field, y_test_in)
        # df = pd.concat((X_train_in, X_test_in))
        # print("X_train_in {}".format(X_train_in.shape))
        # print("X_test_in {}".format(X_test_in.shape))
        # print("df {}".format(df.shape))
        #
        # if 'label_id' in df.columns:
        #     df['label'] = df['label_id']
        #
        # elif 'label' not in df.columns:
        #     raise ValueError("no columns 'label' or 'label_id' exist in input dataframe")
        #
        # df = df[~pd.isnull(df['text'])]
        # print("df {}".format(df.shape))
        #
        # df.loc[:, label_field] = df[label_field].apply(lambda x: str(x) if not pd.isnull(x) else x)
        # df.loc[df[label_field] == ' ', label_field] = None
        #
        # if label_id:
        #     df_labeled = df[df[label_field] == label_id]
        #     df_labeled = pd.concat([df_labeled, df[df[label_field] != label_id].sample(df_labeled.shape[0])])
        #     df_labeled.loc[df_labeled[label_field] != label_id, label_field] = 0
        #     df_labeled = df_labeled[(~pd.isnull(df_labeled[label_field])) & (df_labeled[label_field] != ' ')]
        # else:
        #     df_labeled = df[(~pd.isnull(df[label_field]))]
        # print("df_labeled {}".format(df_labeled.shape))
        # print('Pre-processing text and extracting features...')
        # self.set_preprocessor(pipeline)
        # train_set = df_labeled.loc[df_labeled['document_id'].isin(X_train_in['document_id'])]
        # test_set = df_labeled.loc[df_labeled['document_id'].isin(X_test_in['document_id'])]

        train_preprocessed = self.pre_process(X_train_in, fit=True)
        test_preprocessed = self.pre_process(X_test_in, fit=False)

        # if label_field not in X_train_in.columns:
        #     raise RuntimeError("column '{}' not found".format(label_field))
        # else:
        #     y_train = X_train_in[label_field].values
        #     y_test = X_test_in[label_field].values

        X_train = train_preprocessed.copy()
        X_test = test_preprocessed.copy()
        print('Training the model...')
        self.fit(X_train, y_train_in)

        print('Performance on train set:')
        _, evaluation_text = self.evaluate(X_train, y_train_in)
        result = 'Performance on train set: \n' + evaluation_text

        print('Performance on test set:')
        _, evaluation_text = self.evaluate(X_test, y_test_in)
        result = result + '\nPerformance on test set: \n' + evaluation_text

        df_gold_labels = df[df['user_id'] == 'gold_label']
        y_gold_labels = df_gold_labels[label_field].values
        if len(y_gold_labels)>0:
            X_gold_labels = self.pre_process(df_gold_labels, fit=False)
            print('Performance on gold labels set:')
            _, evaluation_text = self.evaluate(X_gold_labels, y_gold_labels)
            result = result + '\nPerformance on gold labels set: \n' + evaluation_text
        else:
            print('Gold labels do not exist - skipping the evaluation of model performance on them.')

        if run_on_entire_dataset:
            print('Running the model on the entire dataset...')

            columns = ['document_id', label_field, 'user_id', 'prob']

            if bootstrap_iterations > 0:
                print('Bootstrapping...')
            y_aug = df[label_field].copy()
            for i in range(bootstrap_iterations+1):
                # fitting on labeled examples
                has_label = ~pd.isna(y_aug)
                X_labeled = self.pre_process(df.loc[has_label], fit=False)
                self.fit(X_labeled, y_aug[has_label])

                # predict in chunks and (optionally) add bootstrapped labels
                chunk_size = 10000
                n_samples = df.shape[0]
                for chunk_start in tqdm(range(0, n_samples, chunk_size)):
                    chunk_end = min(n_samples, chunk_start + chunk_size)
                    chunk_df = df.iloc[chunk_start:chunk_end]
                    chunk_df.loc[:, label_field] = None
                    y_chunk = df.iloc[chunk_start:chunk_end][label_field]
                    X_chunk = self.pre_process(chunk_df, fit=False)

                    if i < bootstrap_iterations:
                        print('bootstrap iteration ', i, '/', bootstrap_iterations, ' ',
                              [x for x in zip(np.unique(y_aug[has_label], return_counts=True))])

                        # no need to re-fit the model, only predict
                        y_chunk_aug = self.bootstrap(X_chunk, y=y_chunk, th=bootstrap_threshold, fit=False)
                        y_aug.iloc[chunk_start:chunk_end] = y_chunk_aug

                    # write to file only in last iteration
                    if i == bootstrap_iterations:
                        chunk_prediction_df = self.get_prediction_df(X_chunk, y=y_chunk)

                        chunk_prediction_df['document_id'] = df['document_id']
                        chunk_prediction_df['user_id'] = user_id
                        chunk_prediction_df = chunk_prediction_df.rename({'confidence': 'prob'}, axis=1)
                        chunk_prediction_df[label_field] = chunk_prediction_df['prediction']
                        chunk_prediction_df[columns].to_csv(output_filename, index=False, header=True)

        # output_df = pd.DataFrame(columns=columns)
        # output_df.to_csv(output_filename, index=False, header=True, index_label=False)

        print('Saving model weights to file...')
        class_weights = self.important_features
        class_weights_filename = os.path.join(output_folder,
                                              'ml_logistic_regression_weights_{project_id}.csv'.format(project_id=project_id))
        class_weights.to_csv(class_weights_filename, header=True, index=False)

        print('Saving model to a pickle file...')
        model_save_filename = os.path.join(output_folder, 'ml_model_{project_id}.pickle'.format(project_id=project_id))
        self.save(model_save_filename)

        print('Saving model results to a text file...')
        ml_model_results_filename = os.path.join(output_folder, 'ml_model_results_{}.txt'.format(project_id))
        with open(ml_model_results_filename, 'wt') as f:
            f.write(result)

        y_test_pred = self.predict(X_test)
        y_test_pred_proba = self.predict_proba(X_test)

        # # Showing examples of large errors
        # df_labeled.loc[:, 'y_pred'] = self.predict(X)
        # df_labeled.loc[:, 'is_error'] = df_labeled['y_pred']!=df_labeled[label_field]
        # df_labeled.loc[:, 'y_pred_proba'] = np.max(self.predict_proba(X), axis=1)
        # df_labeled.to_csv(output_filename, index=False, header=True, index_label=False)

        # Confusion matrix
        print('Generating confusion matrix...')
        from src.utils.analyze_model import plot_confusion_matrix
        fig = plot_confusion_matrix(y_test_in, y_test_pred, classes=None, normalize=True, title='Normalized confusion matrix - test')
        filename = os.path.join(output_folder, 'confusion_matrix_test_{}.png'.format(project_id))
        fig.savefig(filename)
        plt.clf()

        fig = plot_confusion_matrix(y_train_in, self.predict(X_train), classes=None, normalize=True, title='Normalized confusion matrix - train')
        filename = os.path.join(output_folder, 'confusion_matrix_train_{}.png'.format(project_id))
        fig.savefig(filename)
        plt.clf()

        # Precision-recall curve
        print('Generating the Precision-Recall graph...')
        try:
            fig = plot_precision_recall_curve(y_test_pred_proba, y_test_in)
            filename = os.path.join(output_folder, 'precision_recall_curve_{}.png'.format(project_id))
            fig.savefig(filename)
            plt.clf()
        except ValueError as e:
            print(e)

        # ROC curve
        print('Generating ROC curve...')
        try:
            fig = plot_roc_curve(y_test_pred_proba, y_test_in)
            filename = os.path.join(output_folder, 'roc_curve_{}.png'.format(project_id))
            fig.savefig(filename)
            plt.clf()
        except ValueError as e:
            print(e)

        # Confidence-accuracy graph
        print('Generating the Confidence-Accuracy graph...')
        try:
            fig = plot_confidence_performance(y_test_pred, y_test_pred_proba, y_test_in)
            filename = os.path.join(output_folder, 'confidence_accuracy_graph_{}.png'.format(project_id))
            fig.savefig(filename)
            plt.clf()
        except ValueError as e:
            print(e)

        # Confidence Distribution
        print('Computing distribution of confidence...')
        try:
            ax = pd.Series(np.max(y_test_pred_proba, axis=1)).hist(bins=50)
            plt.xlabel('Confidence'); plt.ylabel('Counts')
            filename = os.path.join(output_folder, 'confidence_distribution_{}.png'.format(project_id))
            plt.gcf().savefig(filename)
            plt.clf()
        except ValueError as e:
            print(e)

        # Generating learning curve
        print('Generating the learning curve...')
        from src.utils.analyze_model import plot_learning_curve_cv
        fig = plot_learning_curve_cv(X_train, y_train_in, estimator=self._model)
        filename = os.path.join(output_folder, 'learning_curve_{}.png'.format(project_id))
        fig.savefig(filename)
        plt.clf()

        # # Run FastText for text classification
        df_labeled_train = df.loc[X_train.index, :]
        df_labeled_test = df.loc[X_test.index, :]

        if RUN_FASTTEXT:
            try:
                print('Running FastText model...')
                import fasttext

                def write_as_fasttext_format(df, filename):
                    with open(filename, 'wt', encoding='utf-8') as f:
                        _ = [f.write('{} __label__{}\n'.format( r['text'].lower().replace('\n', ' '), r['label_id'].replace(' ', '_'))) for i,r in df.iterrows()]

                write_as_fasttext_format(df_labeled_train, output_folder+'/fasttext_train.txt')
                write_as_fasttext_format(df_labeled_test, output_folder+'/fasttext_test.txt')
                classifier = fasttext.train_supervised(output_folder+'/fasttext_train.txt', )
                fasttext_result = classifier.test(output_folder+'/fasttext_test.txt')
                fasttext_pred = classifier.predict([r['text'].lower().replace('\n', ' ') for i, r in df_labeled_test.iterrows()])
                fasttext_pred = [x[0] for x in fasttext_pred]
                # fasttext_pred = ['1' if x == ['__label__1'] else '0' for x in fasttext_pred[0]]

                _, evaluation_text = self.evaluate(X=None, y=df_labeled_test['label_id'].str.replace(' ', '_').values, y_pred=fasttext_pred)
                result += '\nFastText performance on gold labels set: \n' + evaluation_text
            except Exception as e:
                print(e)

        print('Done running the model!')
        return result


def run_model_on_file(input_filename, output_filename, user_id, project_id, label_id=None, method='bow', run_on_entire_dataset=False):
    model = LogisticRegression(verbose=False, class_weight='balanced', random_state=0, penalty='l1', C= 1, solver='liblinear', multi_class='ovr')
    clf = BootstrapClassifier(model=model)
    # pipeline functions are applied sequentially by order of appearance
    pipeline = [('base processing', {'col': 'text', 'new_col': 'processed_text'}),
                ('bag of words', {'col': 'processed_text',
                                  'use_idf': True, 'smooth_idf': True,
                                  'min_df': 2, 'max_df': .9, 'binary': True, 'ngram_range': ngram_range,
                                  'stop_words': 'english', 'strip_accents': 'ascii', 'max_features': 5000}),
                ('drop columns', {'drop_cols': ['label_id', 'text', 'processed_text', 'index', 'source']})]

    result = clf.run_on_file(input_filename, output_filename, user_id, project_id, label_id,
                             pipeline=pipeline, run_on_entire_dataset=run_on_entire_dataset)
    return result


def run_model_on_train_test_split(df, X_train, X_test, y_train, y_test, output_filename, user_id, project_id,
                                  C= 1, label_id=None, method='bow', run_on_entire_dataset=False):
    model = LogisticRegression(verbose=False, class_weight=None, random_state=0, penalty='l1', C=C, solver='liblinear', multi_class='ovr')
    clf = BootstrapClassifier(model=model)
    # pipeline functions are applied sequentially by order of appearance
    pipeline = [('base processing', {'col': 'text', 'new_col': 'processed_text'}),
                ('bag of words', {'col': 'processed_text',
                                  'use_idf': True, 'smooth_idf': True,
                                  'min_df': 2, 'max_df': .9, 'binary': True, 'ngram_range': ngram_range,
                                  'stop_words': 'english', 'strip_accents': 'ascii', 'max_features': 5000}),
                ('drop columns', {'drop_cols': ['label_id', 'text', 'processed_text', 'index', 'source']})]

    result = clf.run_on_train_test(df, X_train, X_test, y_train, y_test, output_filename, user_id, project_id, label_id,
                                   pipeline=pipeline, run_on_entire_dataset=run_on_entire_dataset)
    return result, clf

if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'data'))
    input_file = os.path.join(data_path,'new_data_0030_roy.csv')
    output_file = os.path.join(data_path, 'try_output.csv')
    data = pd.read_csv(input_file)
    label_column_name = "label_id"
    data_column_names = list(data.columns.values)
    data_column_names.remove(label_column_name)
    X_train, X_test, y_train, y_test = train_test_split(data[data_column_names], data[label_column_name], test_size=0.25)
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    result, bootstrap_clf = run_model_on_train_test_split(
        X_train, X_test, y_train, y_test,
        output_filename=output_file,
        user_id=2,
        project_id=998,
        label_id=None,
        method='bow',
        run_on_entire_dataset=False)




import sqlite3
import pickle

import pandas as pd
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from foe_orientation import FOEOrientation
from foe_fingerprint import FOEFingerprint


class FOEResults:
    def __init__(self):
        self.data = {"filename": [], "is_good": [], "is_test": [], "fold": [],
                     "row": [], "column": [], "target": [], "y_out": []}
        self._fold = None

    def append(self, patch, is_test, target, y_out):
        self.data['filename'].append(patch.filename)
        self.data['is_good'].append(patch.fp_type == 1)
        self.data['is_test'].append(is_test)
        self.data['fold'].append(self._fold)
        self.data['row'].append(patch.r)
        self.data['column'].append(patch.c)
        self.data['target'].append(target)
        self.data['y_out'].append(y_out)

    def merge(self, results):
        self.data['filename'].extend(results.data['filename'])
        self.data['is_good'].extend(results.data['is_good'])
        self.data['is_test'].extend(results.data['is_test'])
        self.data['fold'].extend(results.data['fold'])
        self.data['row'].extend(results.data['row'])
        self.data['column'].extend(results.data['column'])
        self.data['target'].extend(results.data['target'])
        self.data['y_out'].extend(results.data['y_out'])

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def from_dataframe(self, df):
        self.data = {}
        for col in df.columns:
            self.data[col] = df[col].to_list()

    def __len__(self):
        return len(self.data['filename'])

    def save(self, filename, table_name='results', metadata=None,
             if_exists='replace'):
        df = self.to_dataframe()
        df['y_out'] = df['y_out'].apply(lambda x: pickle.dumps(x))

        if metadata is not None:
            for k, v in metadata.items():
                df[k] = v

        con = sqlite3.connect(filename)
        df.to_sql(table_name, con, if_exists=if_exists, index=False)
        con.commit()
        con.close()

    def load(self, filename, table_name='results'):
        con = sqlite3.connect(filename)
        df = pd.read_sql('SELECT * FROM {}'.format(table_name), con)
        con.close()

        df['is_good'] = df['is_good'].apply(lambda x: bool(x))
        df['is_test'] = df['is_test'].apply(lambda x: bool(x))
        df['y_out'] = df['y_out'].apply(lambda x: pickle.loads(x))

        self.from_dataframe(df)

    def set_fold(self, fold_id):
        self._fold = fold_id

    def compute_classification_acc(self, fold_id):
        def find_estimated_class(row):
            return np.argmax(row['y_out'])

        def find_gt_class(row):
            return int(row['target'] / np.pi * len(row['y_out']))
        df = self.to_dataframe()
        df = df[(df['fold'] == fold_id)]
        df['est_class_id'] = df.apply(find_estimated_class, axis=1)
        df['gt_class_id'] = df.apply(find_gt_class, axis=1)

        dft = df[(df.is_good) & (~df.is_test)]
        acc_tra_gd = np.mean(dft.est_class_id == dft.gt_class_id)
        dft = df[(~df.is_good) & (~df.is_test)]
        acc_tra_bd = np.mean(dft.est_class_id == dft.gt_class_id)
        dft = df[(df.is_good) & (df.is_test)]
        acc_val_gd = np.mean(dft.est_class_id == dft.gt_class_id)
        dft = df[(~df.is_good) & (df.is_test)]
        acc_val_bd = np.mean(dft.est_class_id == dft.gt_class_id)

        return acc_tra_gd, acc_tra_bd, acc_val_gd, acc_val_bd

    def create_df_rmse(self, estimator):
        def abs_err_sqr(row):
            gt = row['target']
            px = row['y_out']
            ori = estimator(px).item()
            delta_sqr = FOEOrientation.delta_sqr(gt, ori).item()
            return delta_sqr

        df = self.to_dataframe()
        df['abs-err-sqr'] = df.apply(abs_err_sqr, axis=1)
        df = df.groupby(['filename',
                         'fold']).agg({'is_good': 'first',
                                       'is_test': 'first',
                                       'abs-err-sqr': 'mean'}).reset_index()
        df.columns = ['filename', 'fold', 'is_good', 'is_test', 'rmse']
        df['rmse'] = np.sqrt(df['rmse'])
        df = df.groupby(['is_good',
                         'is_test',
                         'fold']).agg({'rmse': 'mean'}).reset_index()
        return df

    def compute_classification_rmse(self, fold_id, estimator=lambda a: a):
        df = self.create_df_rmse(estimator)
        df = df[(df['fold'] == fold_id)]
        rmse_tra_gd = rmse_tra_bd = rmse_val_gd = rmse_val_bd = 0
        df_rmse = df[(df.is_good) & (~df.is_test)].rmse
        if len(df_rmse) > 0:
            rmse_tra_gd = df_rmse.iloc[0]
        df_rmse = df[(~df.is_good) & (~df.is_test)].rmse
        if len(df_rmse) > 0:
            rmse_tra_bd = df_rmse.iloc[0]
        df_rmse = df[(df.is_good) & (df.is_test)].rmse
        if len(df_rmse) > 0:
            rmse_val_gd = df_rmse.iloc[0]
        df_rmse = df[(~df.is_good) & (df.is_test)].rmse
        if len(df_rmse) > 0:
            rmse_val_bd = df_rmse.iloc[0]

        return rmse_tra_gd, rmse_tra_bd, rmse_val_gd, rmse_val_bd

    def plot_hist(self):
        plt.figure()
        plt.hist(self.data['target'][-18030:], 100, alpha=0.5)
        plt.hist(self.data['y_out'][-18030:], 100, alpha=0.5)
        plt.show()

    def analyze(self, fold_id, estimator):
        def estimate(row):
            px = row['y_out']
            ori = estimator(px).item()
            return ori
        df = self.to_dataframe()
        df = df[(df['fold'] == fold_id)]
        df = df[df.is_test]
        df['estimation'] = df.apply(estimate, axis=1)

        RMSE = []
        for filename in df.filename.unique():
            count = len(df[df['filename'] == filename])
            if df[df['filename'] == filename].iloc[0].is_good:
                fp_type = 'good'
            else:
                fp_type = 'bad'
            fp = FOEFingerprint('../datasets/Finger/FOESamples',
                                filename, fp_type)
            est_img = np.zeros((fp.gt.ori.shape))
            tar_img = np.zeros((fp.gt.ori.shape))

            for _, row in df[df['filename'] == filename].iterrows():
                est_img[row['row'], row['column']] = row['estimation']
                tar_img[row['row'], row['column']] = row['target']

            est_img = ndimage.median_filter(est_img, size=1)
            del_img = np.fabs(est_img-tar_img)
            del_img[del_img > np.pi/2.0] = (np.pi -
                                            del_img[del_img > np.pi/2.0])
            del_img_sqr = del_img * del_img
            rmse = np.sqrt(np.sum(del_img_sqr)/count) / np.pi * 180
            RMSE.append(rmse)

            fig = plt.figure()
            plt.title(fp.filename + ' : ' + str(rmse))
            plt.gray()  # show the filtered result in grayscale
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(fp.image)
            im2 = ax2.imshow(del_img, cmap='viridis')
            plt.colorbar(im2, ax=ax2)
            plt.show()
        return RMSE

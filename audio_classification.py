import os

import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
from numpy.random import uniform
import pickle

SR=22050
NUM_ROWS=40
NUM_COLS=int(SR * 5 / 512) + 1

def splitDataFrameList(df, target_column):
    '''
    df = dataframe to split,
    target_column = the column containing a list of values

    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    '''

    def splitListToRows(row):
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            new_rows.append(new_row)

    new_rows = []
    df.reset_index().apply(splitListToRows, axis=1)
    new_index = df.index.names if df.index.names[
                                      0] is not None else df.index.name if df.index.name is not None else 'index'
    new_df = pd.DataFrame(new_rows).set_index(new_index) if len(new_rows) > 0 else pd.DataFrame(new_rows)
    return new_df


def split_df(df, sec=5):
    df['seconds'] = df.duration.apply(lambda x: [(i + 1) * sec for i in range(int(x / sec))]).astype(int)
    return splitDataFrameList(df, 'seconds').reset_index()

def hash_split(df, col='filename', rate=0.7):
    df['h'] = df[col].apply(hash)
    train = df[df.h.apply(lambda x:abs(x)%100/100<rate)]
    test = df[df.h.apply(lambda x:abs(x)%100/100>=rate)]
    return train.drop(columns=['h']), test.drop(columns=['h'])

def even_df(df, target=1000):
    def split(x, t=1000):
        if x == t:
            return [1]
        if x > t:
            return [1] if uniform(0, 1) < t / x else []
        if x < t:
            n = int(t / x)
            return [1] * (n + 1) if uniform(0, 1) < t / x - n else [1] * n
        
    res = df.merge(df.ebird_code.value_counts().to_frame().reset_index().rename(
        columns={'index': 'ebird_code', 'ebird_code': 'cnt'}), on=['ebird_code']).drop(columns=['index'])
    res['split'] = res.cnt.apply(lambda x:split(x,target))
    return splitDataFrameList(res,'split')


class Dataset():

    def __init__(self, path, df, lables=None):
        self.path = path
        self.dump_path = '/home/mor/Downloads/Birds'
        self.mode = 'train' if lables is None else 'predict'
        self.file_col = 'audio_id' if self.mode=='predict' else 'filename'
        self.df = df.sort_values(by=self.file_col)
        self.lables = sorted(df['lable'].unique()) if lables is None else lables
        self.num_rows = NUM_ROWS
        self.num_cols = NUM_COLS
        self.ds = tf.data.Dataset.from_generator(self.iter_train,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=((self.num_rows, self.num_cols, 1), len(self.lables))
                                                 )

    def size(self):
        return len(self.df)

    def get_lable_vec(self, bird):
        res = np.zeros(len(self.lables))
        res[self.lables.index(bird)] = 1
        return res

    def iter_train(self):
        fle = ''
        for ind, row in self.df.iterrows():
            if row[self.file_col] != fle:
                fle = row[self.file_col]
                if self.mode=='train':
                    audio_path = os.path.join(self.path, f"{row['ebird_code']}/{fle}")
                else:
                    audio_path = os.path.join(self.path, f"{fle}.mp3")
                x, sr = librosa.load(audio_path, sr=SR)
            sec = int(row['seconds'])
            xx = x[max(0, sr * (sec - 5)):min(sr * sec, len(x))]
            spec = librosa.feature.mfcc(y=xx, sr=SR, n_mfcc=40)
            for i in range(self.num_cols - spec.shape[1]):
                spec = np.concatenate([spec, spec[:, -1].reshape(spec.shape[0], 1)], axis=1)
            l = [0]*len(self.lables) if self.mode=='predict' else self.get_lable_vec(row['lable'])
            yield spec.reshape(spec.shape[0], spec.shape[1], 1), l

    def iter_np(self):
        fle = ''
        X, Y = None, None
        for ind, row in self.df.iterrows():
            sec = int(row['seconds'])
            l = [0] * len(self.lables) if self.mode == 'predict' else self.get_lable_vec(row['lable'])
            dump = os.path.join(self.dump_path, f"{row[self.file_col]}_{sec}.pkl")
            if os.path.isfile(dump):
                r = pickle.load(open(dump, "rb"))
            else:
                if row[self.file_col] != fle:
                    fle = row[self.file_col]
                    if self.mode == 'train':
                        audio_path = os.path.join(self.path, f"{row['ebird_code']}/{fle}")
                    else:
                        audio_path = os.path.join(self.path, f"{fle}.mp3")
                    x, sr = librosa.load(audio_path, sr=SR)
                xx = x[max(0, sr * (sec - 5)):min(sr * sec, len(x))]
                spec = librosa.feature.mfcc(y=xx, sr=SR, n_mfcc=40)
                for i in range(self.num_cols - spec.shape[1]):
                    spec = np.concatenate([spec, spec[:, -1].reshape(spec.shape[0], 1)], axis=1)
                r = spec.reshape(spec.shape[0], spec.shape[1], 1)
                pickle.dump(r, open(dump, "wb"))
            X = np.array([r]) if X is None else np.append(X, np.array([r]), axis=0)
            Y = np.array([l]) if Y is None else np.append(Y, np.array([l]), axis=0)


def to_np(self):

    i=0
    for ind, row in self.df.iterrows():
        if i%500==0:
            print(i)

        i=i+1
    return X,Y
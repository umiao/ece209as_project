import h5py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
sys.path.append('..')

# Argument data structure used for argument calling. Support member operator.
class Args(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            return AttributeError(name)

    def __setattr__(self, key, value):
        self[key] = value


def window_truncate(feature_vectors, seq_len):
    """
    Truncate time series into fixed-length slices with sliding window.
    :param feature_vectors: Input time series.
    :param seq_len: Length of the serial data input. All aligned with the seq_len of Physionet (48).
    """
    sample_collector = []
    start_index = 0
    n = feature_vectors.shape[0]
    shift_range_lower = seq_len // 2
    shift_range_upper = seq_len
    # All the extracted series are overlapped more or less, in order to enhance robustness.
    while start_index + seq_len < n:
        sample_collector.append(feature_vectors[start_index: start_index + seq_len])
        start_index += random.randint(shift_range_lower, shift_range_upper)
    return np.asarray(sample_collector).astype('float32')


def random_mask(vector, artificial_missing_rate):
    """
    Generate random masks to shed part of the serial data. Used for training set generation.
    :param Vector: Input vector. Expected to be complete.
    :param artificial_missing_rate: Missing rate of manually introduced missing.
    """
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))
    return indices


def add_artificial_mask(X, artificial_missing_rate, set_name):
    """
    Drop some of the data with input mask. This is for testing / evaluating set generation.
    :param X: Input data to be shedded.
    :param artificial_missing_rate: Missing rate of manually introduced missing.
    :param set_name: Specify the dataset we operate. (train / val / test)
    """
    sample_num, seq_len, feature_num = X.shape
    if set_name == 'train':
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(mask, axis=(0, 1))
        data_dict = {
            'X': X,
            'empirical_mean_for_GRUD': empirical_mean_for_GRUD
        }
    else:
        # Generating the test / val set.
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        # X_hat contains artificial missing values
        X_hat[indices_for_holdout] = np.nan
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)
        data_dict = {
            'X': X.reshape([sample_num, seq_len, feature_num]),
            'X_hat': X_hat.reshape([sample_num, seq_len, feature_num]),
            'missing_mask': missing_mask.reshape([sample_num, seq_len, feature_num]),
            'indicating_mask': indicating_mask.reshape([sample_num, seq_len, feature_num])
        }
    return data_dict


def saving_into_h5(saving_dir, data_dict, classification_dataset):
    """
    Save the processed data into h5 format.
    :param saving_dir: Directory to save the data.
    :param data_dict: Dict contains the data of training / testing / val set.
    :param classification_dataset: Mark if the data to be saved is serial imputation / classification data
    """
    def save_each_set(handle, name, data):
        if len(data) == 0:
            # empty dataset passed
            return
        single_set = handle.create_group(name)
        if classification_dataset:
            single_set.create_dataset('labels', data=data['labels'].astype(int))
        single_set.create_dataset('X', data=data['X'].astype(np.float32))
        if name in ['val', 'test']:
            single_set.create_dataset('X_hat', data=data['X_hat'].astype(np.float32))
            single_set.create_dataset('missing_mask', data=data['missing_mask'].astype(np.float32))
            single_set.create_dataset('indicating_mask', data=data['indicating_mask'].astype(np.float32))

    saving_path = os.path.join(saving_dir, 'datasets.h5')
    with h5py.File(saving_path, 'w') as hf:
        hf.create_dataset('empirical_mean_for_GRUD', data=data_dict['train']['empirical_mean_for_GRUD'])
        save_each_set(hf, 'train', data_dict['train'])
        save_each_set(hf, 'val', data_dict['val'])
        save_each_set(hf, 'test', data_dict['test'])
    return


def load_physionet(raw_data_path='Physio2012_mega/mega', outcome_files_dir='Physio2012_mega/',
                   dataset_name = 'physio2012_37feats', saving_path = './generated_datasets',
                   artificial_missing_rate = 0.1, train_frac=0.8,
                   val_frac=0.2):
    """
    Load and preprocess the raw data of physionet and generate h5 dataset file.
    :param raw_data_path: Directory to raw dataset to be processed.
    :param outcome_files_dir: Directory which stores the label of serial data.
    :param dataset_name: Name of dataset.
    :param saving_path: Directory to save the processed data file.
    :param artificial_missing_rate: Rate of manually introduced missingness.
    :param train_frac: Fraction of training set.
    :param val_frac: Fraction of validation set.
    """
    def process_each_set(set_df, all_labels):
        # gene labels, y
        sample_ids = set_df['RecordID'].to_numpy().reshape(-1, 48)[:, 0]
        y = all_labels.loc[sample_ids].to_numpy().reshape(-1, 1)
        # gene feature vectors, X
        set_df = set_df.drop('RecordID', axis=1)
        feature_names = set_df.columns.tolist()
        X = set_df.to_numpy()
        X = X.reshape(len(sample_ids), 48, len(feature_names))
        return X, y, feature_names

    def keep_only_features_to_normalize(all_feats, to_remove):
        for i in to_remove:
            all_feats.remove(i)
        return all_feats

    np.random.seed(26)
    args = Args({'raw_data_path': raw_data_path, 'outcome_files_dir' : outcome_files_dir,
                   'dataset_name' : dataset_name, 'saving_path' : saving_path,
                 'artificial_missing_rate' : artificial_missing_rate, 'train_frac':train_frac,
                 'val_frac':val_frac, 'dataset_name':dataset_name})

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    outcome_files = ['Outcomes-a.txt', 'Outcomes-b.txt', 'Outcomes-c.txt']
    outcome_collector = []
    for o_ in outcome_files:
        outcome_file_path = os.path.join(args.outcome_files_dir, o_)
        with open(outcome_file_path, 'r') as f:
            outcome = pd.read_csv(f)[['In-hospital_death', 'RecordID']]
        outcome = outcome.set_index('RecordID')
        outcome_collector.append(outcome)
    all_outcomes = pd.concat(outcome_collector)

    all_recordID = []
    df_collector = []
    for filename in os.listdir(args.raw_data_path):
        recordID = int(filename.split('.txt')[0])
        with open(os.path.join(args.raw_data_path, filename), 'r') as f:
            df_temp = pd.read_csv(f)
        df_temp['Time'] = df_temp['Time'].apply(lambda x: int(x.split(':')[0]))
        df_temp = df_temp.pivot_table('Value', 'Time', 'Parameter')
        df_temp = df_temp.reset_index()  # take Time from index as a col

        all_recordID.append(recordID)  # only count valid recordID
        if df_temp.shape[0] != 48:
            missing = list(set(range(0, 48)).difference(set(df_temp['Time'])))
            missing_part = pd.DataFrame({'Time': missing})
            df_temp = df_temp.append(missing_part, ignore_index=False, sort=False)
            df_temp = df_temp.set_index('Time').sort_index().reset_index()
        df_temp = df_temp.iloc[:48]  # only take 48 hours, some samples may have more records, like 49 hours
        df_temp['RecordID'] = recordID
        df_temp['Age'] = df_temp.loc[0, 'Age']
        df_temp['Height'] = df_temp.loc[0, 'Height']
        df_collector.append(df_temp)
    df = pd.concat(df_collector, sort=True)
    df = df.drop(['Age', 'Gender', 'ICUType', 'Height'], axis=1)
    df = df.reset_index(drop=True)
    df = df.drop('Time', axis=1)  # dont need Time col

    train_set_ids, test_set_ids = train_test_split(all_recordID, train_size=args.train_frac)
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=args.val_frac)

    all_features = df.columns.tolist()
    feat_no_need_to_norm = ['RecordID']
    feats_to_normalize = keep_only_features_to_normalize(all_features, feat_no_need_to_norm)

    train_set = df[df['RecordID'].isin(train_set_ids)]
    val_set = df[df['RecordID'].isin(val_set_ids)]
    test_set = df[df['RecordID'].isin(test_set_ids)]

    # standardization
    scaler = StandardScaler()
    train_set.loc[:, feats_to_normalize] = scaler.fit_transform(train_set.loc[:, feats_to_normalize])
    val_set.loc[:, feats_to_normalize] = scaler.transform(val_set.loc[:, feats_to_normalize])
    test_set.loc[:, feats_to_normalize] = scaler.transform(test_set.loc[:, feats_to_normalize])

    train_set_X, train_set_y, feature_names = process_each_set(train_set, all_outcomes)
    val_set_X, val_set_y, _ = process_each_set(val_set, all_outcomes)
    test_set_X, test_set_y, _ = process_each_set(test_set, all_outcomes)

    train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, 'train')
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, 'val')
    test_set_dict = add_artificial_mask(test_set_X, args.artificial_missing_rate, 'test')

    train_set_dict['labels'] = train_set_y
    val_set_dict['labels'] = val_set_y
    test_set_dict['labels'] = test_set_y

    processed_data = {
        'train': train_set_dict,
        'val': val_set_dict,
        'test': test_set_dict
    }

    saved_df = df.loc[:, feature_names]

    total_sample_num = 0
    total_positive_num = 0
    for set_name, rec in zip(['train', 'val', 'test'], [train_set_dict, val_set_dict, test_set_dict]):
        total_sample_num += len(rec["labels"])
        total_positive_num += rec["labels"].sum()

    missing_part = np.isnan(saved_df.to_numpy())
    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=True)
    return


def load_beijing_air(file_path='AirQuality/PRSA_Data_20130301-20170228', artificial_missing_rate=0.1,
                     seq_len=48, dataset_name='UCIair_seqlen48_01masked', saving_path='./generated_datasets'):
    """
    Load and preprocess the raw data of Beijing Air Quality and generate h5 dataset file.
    :param file_path: Directory to raw dataset to be processed.
    :param artificial_missing_rate: Rate of manually introduced missingness.
    :param seq_len: Length of truncated serial data.
    :param dataset_name: Name of generated dataset.
    :param saving_path: Directory to save the processed data file.
    """
    args = Args({'file_path': file_path, 'artificial_missing_rate': artificial_missing_rate,
                 'seq_len': seq_len, 'dataset_name': dataset_name,
                 'saving_path': saving_path})

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    df_collector = []
    station_name_collector = []
    file_list = os.listdir(args.file_path)
    for filename in file_list:
        file_path = os.path.join(args.file_path, filename)
        current_df = pd.read_csv(file_path)
        current_df['date_time'] = pd.to_datetime(current_df[['year', 'month', 'day', 'hour']])
        station_name_collector.append(current_df.loc[0, 'station'])
        # remove duplicated date info and wind direction, which is a categorical col
        current_df = current_df.drop(['year', 'month', 'day', 'hour', 'wd', 'No', 'station'], axis=1)
        df_collector.append(current_df)

    date_time = df_collector[0]['date_time']
    df_collector = [i.drop('date_time', axis=1) for i in df_collector]
    df = pd.concat(df_collector, axis=1)
    args.feature_names = [station + '_' + feature
                          for station in station_name_collector
                          for feature in df_collector[0].columns]
    args.feature_num = len(args.feature_names)
    df.columns = args.feature_names

    df['date_time'] = date_time
    unique_months = df['date_time'].dt.to_period('M').unique()
    selected_as_train = unique_months
    train_set = df[df['date_time'].dt.to_period('M').isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, args.feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)

    train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, 'train')

    processed_data = {
        'train': train_set_dict,
        'test' : dict(),
        'val' : dict()
    }
    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    return


def load_electricity(file_path='Electricity/LD2011_2014.txt', artificial_missing_rate=0.1,
                     seq_len=48, dataset_name='electricity_seqlen48_01masked', saving_path='./generated_datasets'):
    """
    Load and preprocess the raw data of Beijing Air Quality and generate h5 dataset file.
    :param file_path: Directory to raw dataset to be processed.
    :param artificial_missing_rate: Rate of manually introduced missingness.
    :param seq_len: Length of truncated serial data.
    :param dataset_name: Name of generated dataset.
    :param saving_path: Directory to save the processed data file.
    """
    args = Args({'file_path': file_path, 'artificial_missing_rate': artificial_missing_rate,
                 'seq_len': seq_len, 'dataset_name': dataset_name,
                 'saving_path': saving_path})

    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    df = pd.read_csv(args.file_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    feature_names = df.columns.tolist()
    feature_num = len(feature_names)
    df['datetime'] = pd.to_datetime(df.index)

    unique_months = df['datetime'].dt.to_period('M').unique()
    selected_as_train = unique_months
    train_set = df[df['datetime'].dt.to_period('M').isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)

    # add missing values in train set manually
    if args.artificial_missing_rate > 0:
        train_set_X_shape = train_set_X.shape
        train_set_X = train_set_X.reshape(-1)
        indices = random_mask(train_set_X, args.artificial_missing_rate)
        train_set_X[indices] = np.nan
        train_set_X = train_set_X.reshape(train_set_X_shape)

    train_set_dict = add_artificial_mask(train_set_X, args.artificial_missing_rate, 'train')

    processed_data = {
        'train': train_set_dict,
        'test': dict(),
        'val': dict()
    }
    train_sample_num = len(train_set_dict["X"])
    total_sample_num = train_sample_num

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    return

if __name__ == '__main__':
    load_physionet()
    load_beijing_air()
    load_electricity()











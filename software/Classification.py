
import os
from datetime import datetime
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

class DataGetter(Dataset):
    def __init__(self, imputed_data, labels):
        self.imputed_data = imputed_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imputed_data[idx].astype('float32')),
            torch.from_numpy(self.labels[idx].astype('float32')),
        )


class LoadData:
    def __init__(self, original_data_path, imputed_data_path, seq_len, feature_num, batch_size=128, num_workers=4):
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers

        with h5py.File(imputed_data_path, 'r') as hf:
            imputed_train_set = hf['imputed_train_set'][:]
            imputed_val_set = hf['imputed_val_set'][:]
            imputed_test_set = hf['imputed_test_set'][:]

        with h5py.File(original_data_path, 'r') as hf:
            train_set_labels = hf['train']['labels'][:]
            val_set_labels = hf['val']['labels'][:]
            test_set_labels = hf['test']['labels'][:]

        self.train_set = DataGetter(imputed_train_set, train_set_labels)
        self.val_set = DataGetter(imputed_val_set, val_set_labels)
        self.test_set = DataGetter(imputed_test_set, test_set_labels)

    def get_loaders(self):
        train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(self.test_set, self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


class RNN(torch.nn.Module):
    def __init__(self, feature_num, rnn_hidden_size, class_num):
        super().__init__()
        self.rnn = torch.nn.LSTM(feature_num, hidden_size=rnn_hidden_size, batch_first=True)
        self.fcn = torch.nn.Linear(rnn_hidden_size, class_num)

    def forward(self, data):
        hidden_states, _ = self.rnn(data)
        logits = self.fcn(hidden_states[:, -1, :])
        prediction_probabilities = torch.sigmoid(logits)
        return prediction_probabilities


def train(model, train_dataloader, val_dataloader, optimizer):
    patience = 20
    current_patience = patience
    best_ROCAUC = 0
    for epoch in range( epochs):
        model.train()
        for idx, data in enumerate(train_dataloader):
            X, y = map(lambda x: x.to( device), data)
            optimizer.zero_grad()
            probabilities = model(X)
            loss = F.binary_cross_entropy(probabilities, y)
            loss.backward()
            optimizer.step()
        model.eval()
        probability_collector, label_collector = [], []
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader):
                X, y = map(lambda x: x.to( device), data)
                probabilities = model(X)
                probability_collector += probabilities.cpu().tolist()
                label_collector += y.cpu().tolist()
        probability_collector = np.asarray(probability_collector)
        label_collector = np.asarray(label_collector)
        classification_metrics = cal_classification_metrics(probability_collector, label_collector)
        if best_ROCAUC < classification_metrics['ROC_AUC']:
            current_patience = patience
            best_ROCAUC = classification_metrics['ROC_AUC']
            saving_path = os.path.join( sub_model_saving,
                                       'model_epoch_{}_ROCAUC_{:.4f}'.format(epoch, best_ROCAUC))
            torch.save(model.state_dict(), saving_path)
        else:
            current_patience -= 1
        if current_patience == 0:
            break
    logger.info('All done. Training finished.')
def setup_logger(log_file_path, log_name, mode='a'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

def cal_classification_metrics(probabilities, labels, pos_label=1, class_num=1):
    if class_num == 1:
        class_predictions = (probabilities >= 0.5).astype(int)
    elif class_num == 2:
        class_predictions = np.argmax(probabilities, axis=1)
    else:
        assert 'args.class_num>2, class need to be specified for precision_recall_fscore_support'
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, class_predictions,
                                                                       pos_label=pos_label, warn_for=())
    precision, recall, f1 = precision[1], recall[1], f1[1]
    precisions, recalls, _ = metrics.precision_recall_curve(labels, probabilities[:, -1], pos_label=pos_label)
    acc_score = metrics.accuracy_score(labels, class_predictions)
    ROC_AUC, fprs, tprs, thresholds = auc_roc(probabilities[:, -1], labels)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        'classification_predictions': class_predictions,
        'acc_score': acc_score, 'precision': precision, 'recall': recall, 'f1': f1,
        'precisions': precisions, 'recalls': recalls, 'fprs': fprs, 'tprs': tprs,
        'ROC_AUC': ROC_AUC, 'PR_AUC': PR_AUC,
    }
    return classification_metrics


def masked_mae_cal(inputs, target, mask):
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(torch.abs(target * mask)) + 1e-9)


def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true=y_test, probas_pred=y_pred)
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds


def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds


def auc_to_recall(recalls, precisions, recall=0.01):
    precisions_mod = precisions.copy()
    ind = np.where(recalls < recall)[0][0] + 1
    precisions_mod[:ind] = 0
    area = metrics.auc(recalls, precisions_mod)
    return area


if __name__ == '__main__':

    root_dir = "" # the dir where you save your model and log
    original_dataset_path = ""# where you store original data
    imputed_dataset_path = ""#where you store the imputation result
    seq_len = 48 #adjust according to the data
    feature_num = 37 # adjust according to imputation results
    rnn_hidden_size = 128
    epochs = 10000
    lr = 0.001
    saved_model_path = ""#only useful in test mode
    test_mode = True
    device = "cuda"
    if  test_mode:
        assert  saved_model_path is not None, 'saved_model_path must be provided in test mode'

    time_now = datetime.now().__format__('%Y-%m-%d_T%H:%M:%S')
    log_saving = os.path.join( root_dir, 'logs')
    model_saving = os.path.join( root_dir, 'models')
    sub_model_saving = os.path.join(model_saving, time_now)
    [os.makedirs(dir_) for dir_ in [model_saving, log_saving,  sub_model_saving] if not os.path.exists(dir_)]
    logger = setup_logger(os.path.join(log_saving, 'log_' + time_now), 'w')
    model = RNN( feature_num,  rnn_hidden_size, 1)
    dataloader = LoadData( original_dataset_path,  imputed_dataset_path,
                                    seq_len,  feature_num, 128)
    train_set_loader, val_set_loader, test_set_loader = dataloader.get_loaders()
    if 'cuda' in  device and torch.cuda.is_available():
        model = model.to( device)
    if not  test_mode:
        logger.info('Start training...')
        optimizer = torch.optim.Adam(model.parameters(),  lr)
        train(model, train_set_loader, val_set_loader, optimizer)
    else:
        logger.info('Start testing...')
        checkpoint = torch.load( saved_model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        probability_collector, label_collector = [], []
        for idx, data in enumerate(test_set_loader):
            X, y = map(lambda x: x.to( device), data)
            probabilities = model(X)
            probability_collector += probabilities.cpu().tolist()
            label_collector += y.cpu().tolist()
        probability_collector = np.asarray(probability_collector)
        label_collector = np.asarray(label_collector)
        classification_metrics = cal_classification_metrics(probability_collector, label_collector)
        for k, v in classification_metrics.items():
            logger.info(f'{k}: {v}')

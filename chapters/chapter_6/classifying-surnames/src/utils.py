import os
import torch
import numpy as np
from sklearn.metrics import f1_score


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}


def compute_accuracy(y_pred, y_target, metric='f1', return_labels_data=False):
    if metric not in ['accuracy', 'f1']:
        raise ValueError('Accuracy metric should be accuracy of f1')

    _, y_pred_indices = y_pred.max(dim=1)
    if metric == 'accuracy':
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        acc = n_correct / len(y_pred_indices)
    elif metric == 'f1':
        acc = f1_score(y_target, y_pred_indices, average='weighted')
    if return_labels_data:
        return acc *100, y_target, y_pred_indices
    return acc * 100

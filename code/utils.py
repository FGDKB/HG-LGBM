from torch_geometric.data import HeteroData
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import torch

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_curves(y_true, y_pred, save_path='curves.png'):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AUPR = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    ndcg = ndcg_score([real_score], [predict_score])
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    roc_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    roc_dot_matrix.T[0] = [0, 0]
    roc_dot_matrix = np.c_[roc_dot_matrix, [1, 1]]

    x_roc = roc_dot_matrix[0].T
    y_roc = roc_dot_matrix[1].T
    auc = 0.5 * (x_roc[1:] - x_roc[:-1]).T * (y_roc[:-1] + y_roc[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    pr_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    pr_dot_matrix.T[0] = [0, 1]
    pr_dot_matrix = np.c_[pr_dot_matrix, [1, 0]]

    x_pr = pr_dot_matrix[0].T
    y_pr = pr_dot_matrix[1].T
    aupr = 0.5 * (x_pr[1:] - x_pr[:-1]).T * (y_pr[:-1] + y_pr[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print(
        ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}, ndcg:{:.4f}'.format(
            auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision, ndcg))

    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision, fpr,
            tpr, precision_list, recall_list], \
        ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision', 'fpr', 'tpr',
         'precision_list', 'recall_list']


def get_name_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def get_hetero_data(disease_features, microbe_features, edge_index, adjacency_matrix, edge_weight=None):
    disease_features = disease_features.to(device)
    microbe_features = microbe_features.to(device)
    edge_index = edge_index.to(device)
    meta_dict = {
        'disease': {'num_nodes': disease_features.shape[0], 'num_features': disease_features.shape[1]},
        'microbe': {'num_nodes': microbe_features.shape[0], 'num_features': microbe_features.shape[1]},
        ('disease', 'e1', 'microbe'): {'edge_index': edge_index, 'edge_weight': edge_weight},
        ('microbe', 'e1', 'disease'): {'edge_index': torch.flip(edge_index, (0,)), 'edge_weight': edge_weight},
    }

    data = HeteroData(meta_dict)
    data['disease'].x = disease_features
    data['microbe'].x = microbe_features
    data[('disease', 'e1', 'microbe')].edge_index = edge_index
    data[('microbe', 'e1', 'disease')].edge_index = torch.flip(edge_index, (0,))

    data['x_dict'] = {ntype: data[ntype].x for ntype in data.node_types}
    edge_index_dict = {}
    for etype in data.edge_types:
        edge_index_dict[etype] = data[etype].edge_index
    data['edge_dict'] = edge_index_dict
    data['edge_matrix'] = adjacency_matrix

    return data.to(device)


def get_data(params):
    import data_loader

    dataset_name = "HMDAD"
    if hasattr(params, 'dataset') and params.dataset:
        dataset_name = params.dataset

    dataset_path = f"./{dataset_name}"
    if dataset_name != "HMDAD" and hasattr(params, 'datapath'):
        dataset_path = params.datapath + "/" + dataset_name

    load_fn = data_loader.get_load_fn(dataset_name)
    adj_matrix, disease_similarity, microbe_similarity, _, _ = load_fn(dataset_path)

    edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)

    edge_index_neg = np.column_stack(np.argwhere(adj_matrix == 0))
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

    num_pos_edges_number = edge_index_pos.shape[1]
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges_number,),
                                               dtype=torch.long)
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]
    edge_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)
    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)

    disease_features = torch.tensor(disease_similarity, dtype=torch.float32)
    microbe_features = torch.tensor(microbe_similarity, dtype=torch.float32)

    data = get_hetero_data(disease_features, microbe_features, edge_index_pos, adj_matrix)
    return data, y, edge_index_all


def plot_multiple_curves(fold_results, model_name='lightgbm', save_path='curves/average_roc_pr_curves.png'):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_title('ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.grid(True, linestyle='--', alpha=0.7)

    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]

    all_tprs = []
    all_fprs = []
    all_precisions = []
    all_recalls = []

    for fold_idx, fold_result in enumerate(fold_results):
        if model_name not in fold_result:
            continue

        model_result = fold_result[model_name]
        if 'best_metrics' not in model_result:
            continue

        metrics = model_result['best_metrics']
        if 'fpr' not in metrics or 'tpr' not in metrics or 'precision_list' not in metrics or 'recall_list' not in metrics:
            continue

        fpr = metrics['fpr'].flatten()
        tpr = metrics['tpr'].flatten()
        all_fprs.append(fpr)
        all_tprs.append(tpr)

        fold_auc = metrics['auc']
        ax1.plot(fpr, tpr, color=colors[fold_idx % 10], alpha=0.6,
                 label=f'Fold {fold_idx + 1} (AUC = {fold_auc:.4f})')

        precision = metrics['precision_list'].flatten()
        recall = metrics['recall_list'].flatten()
        all_precisions.append(precision)
        all_recalls.append(recall)

        fold_prc = metrics['prc']
        ax2.plot(recall, precision, color=colors[fold_idx % 10], alpha=0.6,
                 label=f'Fold {fold_idx + 1} (AUPR = {fold_prc:.4f})')

    if all_fprs and all_tprs:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)

        for fpr, tpr in zip(all_fprs, all_tprs):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)

        mean_tpr /= len(all_tprs)

        mean_auc = auc(mean_fpr, mean_tpr)

        ax1.plot(mean_fpr, mean_tpr, color='red', lw=2,
                 label=f'Average ROC (AUC = {mean_auc:.4f})')

    if all_recalls and all_precisions:
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(mean_recall)

        for recall, precision in zip(all_recalls, all_precisions):
            mean_precision += np.interp(mean_recall, recall, precision, left=1, right=0)

        mean_precision /= len(all_precisions)

        mean_aupr = auc(mean_recall, mean_precision)

        ax2.plot(mean_recall, mean_precision, color='red', lw=2,
                 label=f'Average PR (AUPR = {mean_aupr:.4f})')

    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path

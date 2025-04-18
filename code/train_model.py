import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from models import HGT
from utils import get_metrics
from sklearn.model_selection import KFold
import torch
import copy
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_kf_auc_list_to_txt(kf_auc_list, filename="kf_auc_list.txt"):
    with open(filename, "w") as f:
        for fold_idx, auc_data in enumerate(kf_auc_list):
            f.write(f"Fold {fold_idx + 1}:\n")
            for entry in auc_data:
                if isinstance(entry, np.ndarray):
                    f.write("Array: " + " ".join(map(str, entry)) + "\n")
                else:
                    f.write("Scalar: " + str(entry) + "\n")
            f.write("\n")

def train_model(data,y, edge_index_all, train_idx, test_idx, param, k_number):
    hidden_channels, num_heads, num_layers = param.hidden_channels, param.num_heads, param.num_layers
    epoch_param = param.epochs
    model = HGT(hidden_channels, num_heads=num_heads, num_layers=num_layers, data=data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)
    data_temp = copy.deepcopy(data)
    model.data_temp = data_temp
    model.edge_index_all = edge_index_all
    auc_list = []
    model.train()
    model.param = param
    best_auc = 0.0
    patience = 20
    no_improve = 0
    for epoch in range(1, epoch_param+1):
        optimizer.zero_grad()
        model.pkl_ctl = 'train'
        y_train = y[train_idx].to('cpu').detach().numpy()
        out = model(data_temp, edge_index=edge_index_all.to(device))
        loss = F.binary_cross_entropy_with_logits(out[train_idx].to(device), y[train_idx].to(device))
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if epoch % param.print_epoch == 0:
            model.pkl_ctl='test'
            model.eval()
            with torch.no_grad():
                out = model(data_temp, edge_index=edge_index_all)
                out_pred_s = out[test_idx].to('cpu').detach().numpy()
                out_pred = out_pred_s
                y_true = y[test_idx].to('cpu').detach().numpy()
                auc = roc_auc_score(y_true, out_pred)
                idx = np.arange(y.shape[0])
                if model.best_auc < auc:
                    model.best_auc = auc
                    no_improve = 0
                    model.save_model_state(k_number, train_idx, test_idx,y)
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.other_args['arg_value'])
                auc_idx.append(epoch)
            auc_list.append(auc_idx)
            model.train()
    auc_name.extend(param.other_args['arg_name'])
    return auc_list, auc_name

def CV_train(param, args_tuple=()):
    data, y, edge_index_all = args_tuple
    idx = np.arange(y.shape[0])
    k_number = 1
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)
    kf_auc_list = []
    all_y_true = []
    all_y_pred = []
    for train_idx, test_idx in kf.split(idx):
        print(f'Running fold {k_number} of {k_fold}...')
        auc_idx, auc_name = train_model(data, y, edge_index_all, train_idx, test_idx, param, k_number)
        with open(f'data/model_{k_number}_fold.dict', 'rb') as f:
            data_dict = joblib.load(f)
        all_y_true.extend(data_dict['y_test'])
        all_y_pred.extend(data_dict['y_test_pred'])
        k_number += 1
        kf_auc_list.append(auc_idx)
    from utils import plot_curves
    plot_curves(np.array(all_y_true), np.array(all_y_pred), save_path='Disbiome/curves.png')
    save_kf_auc_list_to_txt(kf_auc_list, "kf_auc_list.txt")
    data_idx = kf_auc_list
    return data_idx, auc_name

def hg_sampling(data, batch_size):
    sampled_nodes = {}
    for node_type in data.node_types:
        node_count = data[node_type].x.size(0)
        sampled_nodes[node_type] = torch.randperm(node_count)[:batch_size]
    sampled_edges = {}
    for edge_type in data.edge_types:
        edges = data[edge_type].edge_index
        mask = (edges[0].isin(sampled_nodes[edge_type[0]]) &
                edges[1].isin(sampled_nodes[edge_type[2]]))
        sampled_edges[edge_type] = edges[:, mask]
    return sampled_nodes, sampled_edges

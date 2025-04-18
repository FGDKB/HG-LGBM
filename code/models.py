from torch_geometric.nn import Linear
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from utils import get_metrics, get_name_list, plot_curves
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomHGTConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata, num_heads=1, dropout=0.2):
        super().__init__()
        
        self.metadata = metadata
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.linear_relations = torch.nn.ModuleDict()
        
        node_types, edge_types = metadata[0], metadata[1]
        
        for edge_type in edge_types:
            src, edge, dst = edge_type
            
            self.linear_relations[f"{src}_{edge}_{dst}_k"] = Linear(-1, out_channels * num_heads)
            self.linear_relations[f"{src}_{edge}_{dst}_q"] = Linear(-1, out_channels * num_heads)
            self.linear_relations[f"{src}_{edge}_{dst}_v"] = Linear(-1, out_channels * num_heads)
            
            self.linear_relations[f"{src}_{edge}_{dst}_a"] = Linear(out_channels * num_heads, num_heads)
        
        self.linear_out = torch.nn.ModuleDict()
        for node_type in node_types:
            self.linear_out[node_type] = Linear(out_channels * num_heads, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        result_dict = {node_type: [] for node_type in x_dict.keys()}
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, edge_type_name, dst_type = edge_type
            
            src_nodes = x_dict[src_type]
            dst_nodes = x_dict[dst_type]
            
            row, col = edge_index
            
            src_features = src_nodes[row]
            dst_features = dst_nodes[col]
            
            k = self.linear_relations[f"{src_type}_{edge_type_name}_{dst_type}_k"](src_features)
            q = self.linear_relations[f"{src_type}_{edge_type_name}_{dst_type}_q"](dst_features)
            v = self.linear_relations[f"{src_type}_{edge_type_name}_{dst_type}_v"](src_features)
            
            k = k.view(-1, self.num_heads, self.out_channels)
            q = q.view(-1, self.num_heads, self.out_channels)
            v = v.view(-1, self.num_heads, self.out_channels)
            
            alpha = (k * q).sum(dim=-1) / (self.out_channels ** 0.5)
            alpha = F.leaky_relu(alpha, 0.2)
            
            alpha = F.softmax(alpha, dim=0)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            alpha = alpha.unsqueeze(-1)
            
            out = v * alpha
            
            out = out.reshape(-1, self.num_heads * self.out_channels)
            
            out_dst = torch.zeros(dst_nodes.size(0), self.num_heads * self.out_channels, device=out.device)
            out_dst.scatter_add_(0, col.view(-1, 1).expand(-1, self.num_heads * self.out_channels), out)
            
            out_dst = self.linear_out[dst_type](out_dst)
            
            result_dict[dst_type].append(out_dst)
        
        for node_type, features_list in result_dict.items():
            if features_list:
                result_dict[node_type] = torch.stack(features_list).mean(dim=0)
            else:
                result_dict[node_type] = x_dict[node_type]
        
        return result_dict


class HGT(torch.nn.Module):

    def __init__(self, hidden_channels, num_heads, num_layers, data):

        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = CustomHGTConv(-1, hidden_channels, data.metadata(), num_heads)
            self.convs.append(conv)

        self.fc = Linear(hidden_channels * 2, 2)
        self.dropout = torch.nn.Dropout(0.5)

        self.save_data = None
        self.edge_index = None
        self.data_temp = None
        self.edg_index_all = None

        self.best_auc = 0.0
        self.param = None

    def forward(self, data, edge_index):

        x_dict_, edge_index_dict = data['x_dict'], data['edge_dict']
        x_dict = x_dict_.copy()

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        all_layers_features = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_layers_features.append(x_dict.copy())

        for node_type in x_dict_.keys():
            x_dict[node_type] = torch.cat([layer[node_type] for layer in all_layers_features], dim=1)

        self.save_data = x_dict
        self.edge_index = edge_index

        m_index = edge_index[0]
        d_index = edge_index[1]

        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])

        scores = Em @ Ed.t()
        predictions = scores[m_index, d_index].unsqueeze(-1)

        return predictions

    def save_model_state(self, kf, train_idx, test_idx, y):
        self._store_indices_and_labels(train_idx, test_idx, y)
        
        self._process_and_save_features(kf)
    
    def _store_indices_and_labels(self, train_idx, test_idx, y):
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.y = y
    
    def _process_and_save_features(self, kf):
        data_concat = torch.concat(
            (self.save_data['n1'][self.edge_index[0]], self.save_data['n2'][self.edge_index[1]]),
            dim=1
        ).cpu().numpy()
        
        train_data_concat = data_concat[self.train_idx]
        test_data_concat = data_concat[self.test_idx]
        
        with torch.no_grad():
            out = self(self.data_temp, self.edg_index_all)
            y_test_pred = out[self.test_idx].cpu().numpy().flatten()
        
        # 准备数据
        train_data = {
            'features': train_data_concat,
            'labels': self.y[self.train_idx].cpu().numpy()
        }
        
        test_data = {
            'features': test_data_concat,
            'labels': self.y[self.test_idx].cpu().numpy(),
            'predictions': y_test_pred
        }
        
        node_embeddings = {
            'Em': self.save_data['n1'].cpu().numpy(),
            'Ed': self.save_data['n2'].cpu().numpy()
        }
        
        # 保存所有数据到文件
        output_path = f'HMDAD/mid_data_HMDAD/{6}nl{kf}kf_best_cat_data.dict'
        save_data = {
            'train_data': train_data['features'],
            'test_data': test_data['features'],
            'y_train': train_data['labels'],
            'y_test': test_data['labels'],
            'y_test_pred': test_data['predictions'],
            'all_data': node_embeddings
        }
        
        joblib.dump(save_data, output_path)
    
    def concat_same_m_d(self, kf):
        self._process_and_save_features(kf)


class ModelSelector:


    def __init__(self, custom_models=None, custom_params=None):
        default_models = {
            'lightgbm': LGBMClassifier()
        }
        
        default_param_grids = {
            'lightgbm': {
                'num_leaves': [25], 
                'learning_rate': [0.15], 
                'max_depth': [-1]
            }
        }
        
        self.models = default_models.copy()
        if custom_models:
            self.models.update(custom_models)
            
        self.param_grids = default_param_grids.copy()
        if custom_params:
            self.param_grids.update(custom_params)
    
    def get_models(self, model_list=[]):

        if not model_list:
            return self.models
        return {key: self.models[key] for key in model_list if key in self.models}

    def train_with_grid_search(self, X_train, y_train, X_test, y_test, models_dict=None):

        if models_dict is None:
            models_dict = self.models

        results = {}

        for model_name, model in models_dict.items():
            print(f"训练模型: {model_name}...")
            param_grid = self.param_grids.get(model_name, {})
            grid = ParameterGrid(param_grid)

            best_score = -1
            best_params = None
            best_metrics = None
            best_model = None
            best_time = float('inf')
            for params in grid:
                print(f"测试参数: {params}")
                model.set_params(**params)


                model.fit(X_train, y_train)

                y_score = model.predict_proba(X_test)[:, 1]


                auc = roc_auc_score(y_test, y_score)

                metrics_values, metrics_names = get_metrics(y_test, y_score)
                metrics_dict = dict(zip(metrics_names, metrics_values))

                print(f"{model_name}模型在参数{params}下的AUC: {auc:.4f}")
                print(f"详细评估指标:")
                print(f"AUC: {metrics_dict['auc']:.4f}")
                print(f"AUPR: {metrics_dict['prc']:.4f}")
                print(f"F1-Score: {metrics_dict['f1_score']:.4f}")
                print(f"Accuracy: {metrics_dict['acc']:.4f}")
                print(f"Recall: {metrics_dict['recall']:.4f}")
                print(f"Specificity: {metrics_dict['specificity']:.4f}")
                print(f"Precision: {metrics_dict['precision']:.4f}")
                print(f"运行时间: {metrics_dict['run_time']:.4f} 秒")

                if auc > best_score:
                    best_score = auc
                    best_params = params
                    best_metrics = metrics_dict
                    best_model = model

            result_dict = {
                'best_score': best_score,
                'best_params': best_params,
                'best_metrics': best_metrics,
                'best_model': best_model,
                'best_run_time': best_time,
                'all_data': {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test
                }
            }

            if best_metrics:
                print(f"\n{model_name}最佳AUC: {best_score:.4f}")
                print(f"最佳参数: {best_params}")
                print(f"最佳运行时间: {best_time:.4f} 秒")

                try:
                    plot_curves(best_metrics['y_true'], best_metrics['y_score'],
                                f"curves/{model_name}_roc_pr_curves.png")
                    print(f"曲线已保存到curves/{model_name}_roc_pr_curves.png")
                except Exception as e:
                    print(f"绘制曲线时出错: {e}")

            results[model_name] = result_dict

        return results

    def calculate_average_performance(self, all_fold_results):

        all_models = set()
        for fold_result in all_fold_results:
            all_models.update(fold_result.keys())

        average_results = {}

        for model_name in all_models:
            model_folds = [fold_result.get(model_name) for fold_result in all_fold_results
                           if fold_result.get(model_name) is not None]

            if not model_folds:
                continue

            metrics_keys = ['auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']
            metrics_values = {key: [] for key in metrics_keys}
            run_times = []

            for fold in model_folds:
                if fold.get('best_metrics'):
                    for key in metrics_keys:
                        if key in fold['best_metrics']:
                            metrics_values[key].append(fold['best_metrics'][key])

                if 'best_run_time' in fold:
                    run_times.append(fold['best_run_time'])
                elif 'best_metrics' in fold and 'run_time' in fold['best_metrics']:
                    run_times.append(fold['best_metrics']['run_time'])

            avg_metrics = {}
            for key in metrics_keys:
                if metrics_values[key]:
                    avg_metrics[key + '_mean'] = np.mean(metrics_values[key])
                    avg_metrics[key + '_std'] = np.std(metrics_values[key])


            average_results[model_name] = avg_metrics

        for model_name, metrics in average_results.items():
            print(f"\n{model_name}模型平均性能:")
            for key, value in metrics.items():
                if '_mean' in key:
                    metric_name = key.replace('_mean', '')
                    print(f"{metric_name.upper()}: {value:.4f} ± {metrics.get(metric_name + '_std', 0):.4f}")

        return average_results

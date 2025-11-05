import numpy as np
import pandas as pd


class FeatureSelectionTree:
    def __init__(self, loss_function, importance_ratio_threshold=0.1, feature_count_threshold=None, max_depth=5,
                 min_samples_split=10):
        self.loss_function = loss_function
        self.importance_ratio_threshold = importance_ratio_threshold
        self.feature_count_threshold = feature_count_threshold
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_importances_ = None
        self.feature_thresholds = None

    def fit(self, X, y, num_classes):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)

        y_onehot = np.eye(num_classes)[y]

        self.feature_thresholds = [np.unique(X[:, i]) for i in range(n_features)]

        nodes = [(np.arange(n_samples), 0)]
        while nodes:
            node, current_depth = nodes.pop()
            if current_depth >= self.max_depth:
                continue

            best_gain, best_split = 0, None

            for feature_idx in range(n_features):
                values = X[node, feature_idx]
                thresholds = self.feature_thresholds[feature_idx]

                thresholds = np.random.choice(thresholds, min(len(thresholds), 10), replace=False)

                for threshold in thresholds:
                    left = node[values <= threshold]
                    right = node[values > threshold]

                    if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                        continue

                    p_parent = y_onehot[node].sum(axis=0) / len(node)
                    p_left = y_onehot[left].sum(axis=0) / len(left)
                    p_right = y_onehot[right].sum(axis=0) / len(right)

                    loss_parent = -np.sum(p_parent * np.log(p_parent + 1e-9))
                    loss_left = -np.sum(p_left * np.log(p_left + 1e-9))
                    loss_right = -np.sum(p_right * np.log(p_right + 1e-9))

                    gain = loss_parent - (
                            len(left) / len(node) * loss_left +
                            len(right) / len(node) * loss_right
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature_idx, threshold, left, right)

            if best_split is None:
                continue

            self.feature_importances_[best_split[0]] += best_gain

            nodes.extend([(best_split[2], current_depth + 1), (best_split[3], current_depth + 1)])

    def select_features(self, X):
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet.")

        max_importance = np.max(self.feature_importances_)
        sorted_importances = np.sort(self.feature_importances_)[::-1]
        threshold_importance = self.importance_ratio_threshold * max_importance
        print("self.feature_importances_: ", self.feature_importances_)
        feature_count_limit = self.feature_count_threshold

        selected_indices = (self.feature_importances_ >= threshold_importance)
        print("selected_indices: ", selected_indices)

        return X[:, selected_indices], selected_indices


def cross_entropy(y_onehot):
    p = np.clip(y_onehot.mean(axis=0), 1e-9, 1 - 1e-9)
    return -np.sum(p * np.log(p))


root_path = "OFFICE31/"
source1_name = "bd_zhmk_92.csv"
source2_name = "bd_dhys_92.csv"
source3_name = "tj_zgjy_92.csv"

source2_path = root_path + source2_name
df_dhys = pd.read_csv(source2_path, encoding='gbk')

source_data = df_dhys.values[:, 1:]
source_label = df_dhys.values[:, 0] - 1
source_label = source_label.astype(int)

X = source_data
y = source_label
num_classes = 3

tree = FeatureSelectionTree(loss_function=cross_entropy, importance_ratio_threshold=0.05, feature_count_threshold=10)
tree.fit(X, y, num_classes)

print("Feature Importances:", tree.feature_importances_)

X_selected, selected_indices = tree.select_features(X)
print("Selected Features:\n", X.shape, X_selected.shape)



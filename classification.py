import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.n_classes = None
        self.class_labels = None

    def fit(self, X, y):
        # Convert labels to numerical format
        self.class_labels = np.unique(y)
        self.n_classes = len(self.class_labels)
        # Create mapping from string labels to integers
        label_to_num = {label: i for i, label in enumerate(self.class_labels)}
        y_numerical = np.array([label_to_num[label] for label in y])
        
        self.tree = self._grow_tree(X, y_numerical, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            leaf_value = self.class_labels[np.argmax(np.bincount(y))]
            return {'type': 'leaf', 'value': leaf_value}

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:  # No valid split found
            leaf_value = self.class_labels[np.argmax(np.bincount(y))]
            return {'type': 'leaf', 'value': leaf_value}

        # Create child splits
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Grow child trees
        left_subtree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs
                
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue

                gain = self._information_gain(y, y[left_idxs], y[right_idxs])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        weight_l = len(left_child) / len(parent)
        weight_r = len(right_child) / len(parent)
        
        gain = self._entropy(parent) - (
            weight_l * self._entropy(left_child) + 
            weight_r * self._entropy(right_child)
        )
        
        return gain

    def _entropy(self, y):
        hist = np.bincount(y, minlength=self.n_classes)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])
    
class DecisionTreeWithPruning(DecisionTreeClassifier):
    def __init__(self, max_depth=None, prune=False):
        super().__init__(max_depth)
        self.prune = prune
        self.label_to_num = None  # Add this to store the mapping

    def fit(self, X, y, X_val=None, y_val=None):
        # Convert labels to numerical format
        self.class_labels = np.unique(y)
        self.n_classes = len(self.class_labels)
        self.label_to_num = {label: i for i, label in enumerate(self.class_labels)}  # Store the mapping
        y_numerical = np.array([self.label_to_num[label] for label in y])
        
        self.tree = self._grow_tree(X, y_numerical, depth=0)
        
        # Perform pruning if enabled and validation set is provided
        if self.prune and X_val is not None and y_val is not None:
            y_val_numerical = np.array([self.label_to_num[label] for label in y_val])
            self._prune_tree(self.tree, X_val, y_val_numerical)
        
        return self

    def _prune_tree(self, node, X_val, y_val):
        # Base case: if node is a leaf, return
        if node['type'] == 'leaf':
            return

        # If either child is a node, recursively prune them first
        if node['left']['type'] == 'node':
            self._prune_tree(node['left'], X_val, y_val)
        if node['right']['type'] == 'node':
            self._prune_tree(node['right'], X_val, y_val)

        # Get predictions before pruning
        original_predictions = []
        for x in X_val:
            prediction = self._traverse_tree(x, node)
            # Convert string prediction to numerical
            original_predictions.append(self.label_to_num[prediction])
        
        original_accuracy = np.mean([pred == label for pred, label in zip(original_predictions, y_val)])

        # Get majority class from current node's predictions
        majority_class_idx = np.argmax(np.bincount(original_predictions))
        majority_class = self.class_labels[majority_class_idx]
        
        # Test accuracy if we prune this node
        pruned_predictions = [self.label_to_num[majority_class]] * len(y_val)
        pruned_accuracy = np.mean([pred == label for pred, label in zip(pruned_predictions, y_val)])

        # If pruning improves or maintains accuracy, prune the node
        if pruned_accuracy >= original_accuracy:
            node['type'] = 'leaf'
            node['value'] = majority_class
            node.pop('feature', None)
            node.pop('threshold', None)
            node.pop('left', None)
            node.pop('right', None)

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Only pass max_depth to DecisionTreeClassifier
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth
            )
            # Create bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self

    def predict(self, X):
        # Get predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Perform majority voting for each sample
        predictions = []
        for sample_preds in tree_preds.T:
            counter = Counter(sample_preds)
            majority_class = counter.most_common(1)[0][0]
            predictions.append(majority_class)
        
        return np.array(predictions)

class AdvancedDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, max_leaf_nodes=None, oversample=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.tree = None
        self.n_classes = None
        self.class_labels = None
        self.oversample = oversample
        self.n_leaf_nodes = 0

    def _perform_oversampling(self, X, y):
        """
        Performs random oversampling of minority classes to balance the dataset
        """
        # Get class counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_count = np.max(class_counts)
        
        # Store indices for each class
        class_indices = [np.where(y == c)[0] for c in unique_classes]
        
        # Oversample each minority class
        X_resampled = []
        y_resampled = []
        
        for _, indices in enumerate(class_indices):
            # Number of samples needed
            n_samples = max_count - len(indices)
            
            # Add original samples
            X_resampled.append(X[indices])
            y_resampled.append(y[indices])
            
            if n_samples > 0:
                # Randomly sample with replacement
                resampled_indices = np.random.choice(indices, size=n_samples, replace=True)
                X_resampled.append(X[resampled_indices])
                y_resampled.append(y[resampled_indices])
        
        return np.vstack(X_resampled), np.concatenate(y_resampled)

    def fit(self, X, y):
        # Convert labels to numerical format
        self.class_labels = np.unique(y)
        self.n_classes = len(self.class_labels)
        # Create mapping from string labels to integers
        label_to_num = {label: i for i, label in enumerate(self.class_labels)}
        y_numerical = np.array([label_to_num[label] for label in y])
        
        # Perform oversampling if enabled
        if self.oversample:
            X, y_numerical = self._perform_oversampling(X, y_numerical)
        
        self.tree = self._grow_tree(X, y_numerical, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_labels = len(np.unique(y))
        n_samples = len(y)

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split or \
           (self.max_leaf_nodes is not None and self.n_leaf_nodes >= self.max_leaf_nodes):
            leaf_value = self.class_labels[np.argmax(np.bincount(y))]
            self.n_leaf_nodes += 1
            return {'type': 'leaf', 'value': leaf_value}

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:  # No valid split found
            leaf_value = self.class_labels[np.argmax(np.bincount(y))]
            self.n_leaf_nodes += 1
            return {'type': 'leaf', 'value': leaf_value}

        # Create child splits
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs

        # Check if split creates children with minimum samples
        if np.sum(left_idxs) < self.min_samples_leaf or \
           np.sum(right_idxs) < self.min_samples_leaf:
            leaf_value = self.class_labels[np.argmax(np.bincount(y))]
            self.n_leaf_nodes += 1
            return {'type': 'leaf', 'value': leaf_value}
        
        # Grow child trees
        left_subtree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        
        # Determine number of features to consider
        if self.max_features is None:
            n_features_to_consider = n_features
        else:
            n_features_to_consider = min(self.max_features, n_features)
        
        # Randomly select features to consider
        features = np.random.choice(n_features, n_features_to_consider, replace=False)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs
                
                # Check if split creates children with minimum samples
                if np.sum(left_idxs) < self.min_samples_leaf or \
                   np.sum(right_idxs) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, y[left_idxs], y[right_idxs])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        weight_l = len(left_child) / len(parent)
        weight_r = len(right_child) / len(parent)
        
        gain = self._entropy(parent) - (
            weight_l * self._entropy(left_child) + 
            weight_r * self._entropy(right_child)
        )
        
        return gain

    def _entropy(self, y):
        hist = np.bincount(y, minlength=self.n_classes)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

class AdvancedRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 max_features=None, oversample=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.oversample = oversample
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = AdvancedDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                oversample=self.oversample
            )
            # Create bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # Get predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Perform majority voting for each sample
        predictions = []
        for sample_preds in tree_preds.T:
            # Count occurrences of each class
            counter = {}
            for pred in sample_preds:
                counter[pred] = counter.get(pred, 0) + 1
            # Get the class with maximum votes
            majority_class = max(counter.items(), key=lambda x: x[1])[0]
            predictions.append(majority_class)
        
        return np.array(predictions)
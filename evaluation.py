import numpy as np
from classification import DecisionTreeClassifier, AdvancedDecisionTreeClassifier, RandomForestClassifier, AdvancedRandomForestClassifier, DecisionTreeWithPruning
from get_data import load_dataset
from numpy.random import default_rng
import matplotlib.pyplot as plt
from collections import Counter

def confusion_matrix(y_gold, y_prediction, class_labels=None):
    
    if class_labels is None: 
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
    
    confusion = np.zeros((len(class_labels), len(class_labels)))

    class_label_to_index = {label: idx for idx, label in enumerate(class_labels)}

    for true_label, pred_label in zip(y_gold, y_prediction):
        true_idx = class_label_to_index[true_label]
        pred_idx = class_label_to_index[pred_label]
        confusion[true_idx, pred_idx] += 1

    return confusion


def calc_accuracy(confusion_matrix):
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

def calc_recall(y_true, y_prediction):
    confusion = confusion_matrix(y_true, y_prediction)
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)

    return (r, macro_r)

def calc_precision(y_true, y_prediction):
    confusion = confusion_matrix(y_true, y_prediction)
    p = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)
    
    return (p, macro_p)

def calc_F1(y_gold, y_prediction):

    (precisions, _) = calc_precision(y_gold, y_prediction)
    (recalls, _) = calc_recall(y_gold, y_prediction)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)
    
    return (f, macro_f)

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
       
        test_indices = split_indices[k]

        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds

# Individual Testing Of Different Hyperparameters
def trial_max_depth(train_data, train_label, test_data, test_label, max_depth_list):
    
    acc_list = []
    for depth in max_depth_list:
        d_full = AdvancedDecisionTreeClassifier(max_depth=depth, oversample=True)
        d_full.fit(train_data, train_label)
        acc = calc_accuracy(confusion_matrix(test_label, d_full.predict(test_data)))
        acc_list.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(max_depth_list, acc_list, marker='o', linestyle='-')
    plt.xlabel("Max Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Max Depth of Decision Tree")
    plt.grid(True)
    plt.show()

def trial_min_splits(train_data, train_label, test_data, test_label, min_splits_list):
    
    acc_list = []
    for split in min_splits_list:
        d_full = AdvancedDecisionTreeClassifier(max_depth=18, min_samples_split=split, max_features=15, oversample=True)
        d_full.fit(train_data, train_label)
        acc = calc_accuracy(confusion_matrix(test_label, d_full.predict(test_data)))
        acc_list.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(min_splits_list, acc_list, marker='o', linestyle='-')
    plt.xlabel("Max Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Min Sample Splits Parameter of Decision Tree")
    plt.grid(True)
    plt.show()

def trial_max_features(train_data, train_label, test_data, test_label, max_features_list):
    
    acc_list = []
    for feature in max_features_list:
        d_full = AdvancedDecisionTreeClassifier(max_depth=18, min_samples_split=3, max_features=feature, oversample=True)
        d_full.fit(train_data, train_label)
        acc = calc_accuracy(confusion_matrix(test_label, d_full.predict(test_data)))
        acc_list.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(max_features_list, acc_list, marker='o', linestyle='-')
    plt.xlabel("Max Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Min Sample Splits Parameter of Decision Tree")
    plt.grid(True)
    plt.show()

def trial_min_samples_leaf(train_data, train_label, test_data, test_label, min_samples_leaf_list):
    
    acc_list = []
    for no in min_samples_leaf_list:
        d_full = AdvancedDecisionTreeClassifier(max_depth=18, min_samples_split=3, min_samples_leaf=no, max_features=15, oversample=True)
        d_full.fit(train_data, train_label)
        acc = calc_accuracy(confusion_matrix(test_label, d_full.predict(test_data)))
        acc_list.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(min_samples_leaf_list, acc_list, marker='o', linestyle='-')
    plt.xlabel("Max Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Min Sample Splits Parameter of Decision Tree")
    plt.grid(True)
    plt.show()


def trial_max_leaf_nodes(train_data, train_label, test_data, test_label, max_leaf_nodes_list):
    
    acc_list = []
    for no in max_leaf_nodes_list:
        d_full = AdvancedDecisionTreeClassifier(max_depth=18, min_samples_split=3, max_features=15, max_leaf_nodes=no, oversample=True)
        d_full.fit(train_data, train_label)
        acc = calc_accuracy(confusion_matrix(test_label, d_full.predict(test_data)))
        acc_list.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(max_leaf_nodes_list, acc_list, marker='o', linestyle='-')
    plt.xlabel("Max Depth of Decision Tree")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Min Sample Splits Parameter of Decision Tree")
    plt.grid(True)
    plt.show()

def calculate_metrics(all_k_folds, data, labels, val_data, val_labels):
    accuracy = []
    val_accuracy = []
    precisions = np.empty((0,6), dtype=float)
    macro_ps = []
    predictions_list = []
    recall_list = []
    macro_rs = []
    recalls = np.empty((0,6), dtype=float)
    f1_list = []
    macro_fs = []
    f1s = np.empty((0,6), dtype=float)
    
    # Dictionary to store predictions for each sample index
    indexed_predictions = {}
    
    # Initialise the Decision Tree model
    d = AdvancedDecisionTreeClassifier(max_depth=14, min_samples_split=3, max_features=15, oversample=True)
    
    for fold in all_k_folds:
        train_indices, test_indices = fold
        
        # Get training and test data based on the indices
        k_train_data = data[train_indices]
        k_train_label = labels[train_indices]
        k_test_data = data[test_indices]
        k_test_label = labels[test_indices]
        
        # Train the model on the current fold
        d.fit(k_train_data, k_train_label)
        
        # Validate on the validation set
        val_predictions = d.predict(val_data)
        val_confusion = confusion_matrix(val_labels, val_predictions)
        val_accuracy.append(calc_accuracy(val_confusion))
        
        # Make predictions on the test set
        test_predictions = d.predict(k_test_data)
        predictions_list.append(test_predictions)
        
        # Store predictions with their original indices
        for idx, pred in zip(test_indices, test_predictions):
            if idx not in indexed_predictions:
                indexed_predictions[idx] = []
            indexed_predictions[idx].append(pred)
        
        # Create confusion matrix for test set
        test_confusion = confusion_matrix(k_test_label, test_predictions)
        
        # Calculate accuracy from confusion matrix
        accuracy.append(calc_accuracy(test_confusion))
        
        # Get precisions
        precision_list, macro_p = calc_precision(k_test_label, test_predictions)
        precisions = np.vstack([precisions, precision_list])
        macro_ps.append(macro_p)
        
        # Get Recall
        recall_list, macro_r = calc_recall(k_test_label, test_predictions)
        recalls = np.vstack([recalls, recall_list])
        macro_rs.append(macro_r)
        
        # Get F1 scores
        f1_list, macro_f = calc_F1(k_test_label, test_predictions)
        f1s = np.vstack([f1s, f1_list])
        macro_fs.append(macro_f)
    
    # Mode calculation
    modes = []
    for i in range(len(data)):
        if i in indexed_predictions:
            sample_predictions = indexed_predictions[i]
            counter = Counter(sample_predictions)
            mode = counter.most_common(1)[0][0]
            modes.append(mode)
        else:
            modes.append(None)
    
    mode_confusion = confusion_matrix(labels, modes)
    mode_acc = calc_accuracy(mode_confusion)
    
    # Return all metrics including validation accuracy
    return (np.mean(accuracy), np.std(accuracy), 
            np.mean(val_accuracy), np.std(val_accuracy),
            np.mean(macro_ps), np.mean(precisions, axis=0),
            np.mean(macro_rs), np.mean(recalls, axis=0),
            np.mean(macro_fs), np.mean(f1s, axis=0),
            mode_acc)

def print_tree_test(k = 10, train_data=None, data_labels=None, val_data=None, val_labels=None):

    if(train_data is None or data_labels is None or val_data is None or val_labels is None):
        print("Please provide training data and labels")
        return
    
    instances = len(train_data)
    folds = train_test_k_fold(k, instances, default_rng())
    accuracy, stdev, val_accuracy, val_stdev, macro_p, precision_avg, macro_r, recall_avg, macro_f, f1_avg, mode_acc = calculate_metrics(folds, train_data, data_labels, val_data, val_labels)

    # Evaluation
    print("Evaluation of model:")

    # Accuracy
    print("Accuracy: ", accuracy)
    print("Validation Accuracy: ", val_accuracy)

    # Mode Accuracy
    print("Mode Accuracy: ", mode_acc,"\n")

    # Precision
    print("Precision for each class: ", precision_avg)
    print("Macro Precision: ", macro_p, "\n")

    # Recall
    print("Recall for each class: ", recall_avg)
    print("Marco Recall: ", macro_r, "\n")

    # F1 Scores
    print("F1 scores for each class: ", f1_avg)
    print("Marco F1 Scores: ", macro_f, "\n")

    # Standard Deviation
    print("Stdev: ", stdev)
    print("Validation Stdev: ", val_stdev)

def evaluate_random_forest(train_data, train_labels, val_data, val_labels, k=10):
    """
    Evaluate Random Forest using k-fold cross validation
    """
    instances = len(train_data)
    folds = train_test_k_fold(k, instances, default_rng())
    
    # Lists to store metrics
    accuracy = []
    val_accuracy = []
    predictions_list = []
    
    # Dictionary to store predictions for each sample index
    indexed_predictions = {}
    
    # Initialise the Random Forest model with default parameters
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for fold in folds:
        train_indices, test_indices = fold
        
        # Get training and test data for this fold
        k_train_data = train_data[train_indices]
        k_train_label = train_labels[train_indices]
        k_test_data = train_data[test_indices]
        k_test_label = train_labels[test_indices]
        
        # Train the model
        rf.fit(k_train_data, k_train_label)
        
        # Validate on validation set
        val_predictions = rf.predict(val_data)
        val_confusion = confusion_matrix(val_labels, val_predictions)
        val_accuracy.append(calc_accuracy(val_confusion))
        
        # Test on test set
        test_predictions = rf.predict(k_test_data)
        predictions_list.append(test_predictions)
        
        # Store predictions with their original indices
        for idx, pred in zip(test_indices, test_predictions):
            if idx not in indexed_predictions:
                indexed_predictions[idx] = []
            indexed_predictions[idx].append(pred)
        
        # Calculate accuracy for this fold
        test_confusion = confusion_matrix(k_test_label, test_predictions)
        accuracy.append(calc_accuracy(test_confusion))
    
    # Calculate mode predictions
    modes = []
    for i in range(len(train_data)):
        if i in indexed_predictions:
            sample_predictions = indexed_predictions[i]
            counter = Counter(sample_predictions)
            mode = counter.most_common(1)[0][0]
            modes.append(mode)
        else:
            modes.append(None)
    
    # Calculate final metrics
    mode_confusion = confusion_matrix(train_labels, modes)
    mode_acc = calc_accuracy(mode_confusion)
    
    # Print results
    print("\nRandom Forest Evaluation Results:")
    print(f"Average Accuracy: {np.mean(accuracy):.4f} (±{np.std(accuracy):.4f})")
    print(f"Average Validation Accuracy: {np.mean(val_accuracy):.4f} (±{np.std(val_accuracy):.4f})")
    print(f"Mode Accuracy: {mode_acc:.4f}")
    
    return np.mean(accuracy), np.std(accuracy), np.mean(val_accuracy), np.std(val_accuracy), mode_acc

if __name__ == "__main__":
    # Load data for train full
    train_data_full, train_labels_full = load_dataset("data/train_full.txt")
    # Uncomment one at a time for different datasets
    train_data, train_labels = load_dataset("data/train_full.txt")
    # train_data_noisy, train_labels_noisy = load_dataset("data/train_noisy.txt")
    val_data, val_labels = load_dataset("data/validation.txt")
    test_data, test_labels = load_dataset("data/test.txt")
    
    # Test the decision tree on full dataset
    print("Starting Decision Tree Evaluation (Full Dataset)...")
    print_tree_test(
        k=10, 
        train_data=train_data_full, 
        data_labels=train_labels_full,
        val_data=val_data, 
        val_labels=val_labels
    )

    print("\n" + "="*50 + "\n")  # Separator for clarity

    train_data, train_labels = train_data_full, train_labels_full
    #train_data, train_labels = train_data_noisy, train_labels_noisy

    print("Starting Random Forest Evaluation...")
    train_acc, train_std, val_acc, val_std, mode_acc = evaluate_random_forest(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        k=10
    )

    # Evaluate on test set
    print("\nTesting on Test Dataset:")
    rf = AdvancedRandomForestClassifier(
        n_estimators=100, 
        max_depth=14, 
        min_samples_split=3, 
        max_features=15, 
        oversample=True
    )
    rf.fit(train_data, train_labels)
    test_predictions = rf.predict(test_data)
    test_confusion = confusion_matrix(test_labels, test_predictions)

    # Generate Random Forest Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(test_confusion, interpolation='nearest', cmap='Blues')
    plt.title('Advanced Random Forest Confusion Matrix (Train Full)')
    plt.colorbar()

    num_classes = 6
    class_labels = ['A', 'C', 'E', 'G', 'O', 'Q']

    plt.xticks(np.arange(num_classes), [f'{class_labels[i]}' for i in range(num_classes)])
    plt.yticks(np.arange(num_classes), [f'{class_labels[i]}' for i in range(num_classes)])

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(test_confusion[i, j]), ha='center', va='center', color='black')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()
from evaluation import confusion_matrix, calc_accuracy, calc_precision, calc_recall, calc_F1
from get_data import load_dataset
from classification import (DecisionTreeClassifier, AdvancedDecisionTreeClassifier, 
                          RandomForestClassifier, AdvancedRandomForestClassifier)
import numpy as np

def evaluate_model(model_name, predictions, test_labels):
    cm = confusion_matrix(test_labels, predictions)
    accuracy = calc_accuracy(cm)
    precision_per_class, macro_p = calc_precision(test_labels, predictions)
    recall_per_class, macro_r = calc_recall(test_labels, predictions)
    f1_per_class, macro_f = calc_F1(test_labels, predictions)
    
    print(f"\n=== {model_name} ===")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision per class: {precision_per_class}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Recall per class: {recall_per_class}")
    print(f"Macro Recall: {macro_r:.4f}")
    print(f"F1 Score per class: {f1_per_class}")
    print(f"Macro F1 Score: {macro_f:.4f}")
    
    return accuracy, macro_p, macro_r, macro_f

def compare_classifiers(train_data, train_labels, test_data, test_labels, dataset_name):
    print(f"\n{'='*20} Results for {dataset_name} {'='*20}")
    
    # Part 2 Basic Decision Tree (max_depth=18 as in main.py)
    basic_dt = DecisionTreeClassifier(max_depth=18)
    basic_dt.fit(train_data, train_labels)
    basic_preds = basic_dt.predict(test_data)
    basic_metrics = evaluate_model("Basic Decision Tree (Part 2)", basic_preds, test_labels)
    
    # Advanced Decision Tree with optimised parameters
    advanced_dt = AdvancedDecisionTreeClassifier(
        max_depth=14,
        min_samples_split=3,
        max_features=15,
        oversample=True
    )
    advanced_dt.fit(train_data, train_labels)
    advanced_preds = advanced_dt.predict(test_data)
    advanced_metrics = evaluate_model("Advanced Decision Tree (optimised)", advanced_preds, test_labels)
    
    # Basic Random Forest
    basic_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=18  # Same as basic decision tree for fair comparison
    )
    basic_rf.fit(train_data, train_labels)
    basic_rf_preds = basic_rf.predict(test_data)
    basic_rf_metrics = evaluate_model("Basic Random Forest", basic_rf_preds, test_labels)
    
    # Advanced Random Forest with optimised parameters
    advanced_rf = AdvancedRandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=3,
        max_features=15,
        oversample=True
    )
    advanced_rf.fit(train_data, train_labels)
    advanced_rf_preds = advanced_rf.predict(test_data)
    advanced_rf_metrics = evaluate_model("Advanced Random Forest (optimised)", advanced_rf_preds, test_labels)
    
    return basic_metrics, advanced_metrics, basic_rf_metrics, advanced_rf_metrics

if __name__ == "__main__":
    # Load all datasets
    train_full, train_full_labels = load_dataset('data/train_full.txt')
    train_noisy, train_noisy_labels = load_dataset('data/train_noisy.txt')
    test_data, test_labels = load_dataset('data/test.txt')
    
    # Test on train_full.txt
    full_basic, full_advanced, full_basic_rf, full_rf = compare_classifiers(
        train_full, train_full_labels, test_data, test_labels, "train_full.txt"
    )
    
    # Test on train_noisy.txt
    noisy_basic, noisy_advanced, noisy_basic_rf, noisy_rf = compare_classifiers(
        train_noisy, train_noisy_labels, test_data, test_labels, "train_noisy.txt"
    )
    
    # Print comparative analysis
    print("\n" + "="*50)
    print("Comparative Analysis:")
    print("\nAccuracy Comparison:")
    print(f"{'Model':<30} {'train_full.txt':<15} {'train_noisy.txt':<15}")
    print("-"*60)
    print(f"{'Basic Decision Tree':<30} {full_basic[0]:<15.4f} {noisy_basic[0]:<15.4f}")
    print(f"{'Advanced Decision Tree':<30} {full_advanced[0]:<15.4f} {noisy_advanced[0]:<15.4f}")
    print(f"{'Basic Random Forest':<30} {full_basic_rf[0]:<15.4f} {noisy_basic_rf[0]:<15.4f}")
    print(f"{'Advanced Random Forest':<30} {full_rf[0]:<15.4f} {noisy_rf[0]:<15.4f}")
    
    print("\nMacro F1 Score Comparison:")
    print(f"{'Model':<30} {'train_full.txt':<15} {'train_noisy.txt':<15}")
    print("-"*60)
    print(f"{'Basic Decision Tree':<30} {full_basic[3]:<15.4f} {noisy_basic[3]:<15.4f}")
    print(f"{'Advanced Decision Tree':<30} {full_advanced[3]:<15.4f} {noisy_advanced[3]:<15.4f}")
    print(f"{'Basic Random Forest':<30} {full_basic_rf[3]:<15.4f} {noisy_basic_rf[3]:<15.4f}")
    print(f"{'Advanced Random Forest':<30} {full_rf[3]:<15.4f} {noisy_rf[3]:<15.4f}")
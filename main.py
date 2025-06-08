import numpy as np
from classification import DecisionTreeClassifier
from get_data import load_dataset
from evaluation import confusion_matrix, calc_accuracy, print_tree_test
from improvement import train_and_predict


if __name__ == "__main__":

    #Part 1

    # Load the all of the datasets
    t_full, t_full_label = load_dataset('data/train_full.txt')
    t_sub, t_sub_label = load_dataset('data/train_sub.txt')
    t_noisy, t_noisy_label = load_dataset('data/train_noisy.txt')
    t_data, t_label = load_dataset('data/test.txt')
    val_data, val_label = load_dataset('data/validation.txt')
    
    #Part 2

    #Build Decision Tree
    dt = DecisionTreeClassifier(max_depth=18)
    dt.fit(t_full, t_full_label)
    predictions = dt.predict(t_data)
    print(confusion_matrix(t_label, predictions))
    print(calc_accuracy(confusion_matrix(t_label, predictions)))

    #Part 3

    print_tree_test(10, t_full, t_full_label, val_data, val_label) #prints the tree

    #Part 4

    files = ['data/train_full.txt', 'data/train_sub.txt', 'data/train_noisy.txt']

    for file in files:
        print(f"Testing on file: {file}")
        x_data, y_data = load_dataset(file)
        predictions = train_and_predict(x_data, y_data, t_data)
        print(confusion_matrix(t_label, predictions))
        print(calc_accuracy(confusion_matrix(t_label, predictions)))
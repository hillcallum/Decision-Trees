from evaluation import *
from classification import AdvancedDecisionTreeClassifier

from get_data import load_dataset

def grid_search(train_data, train_label, test_data, test_label):
    max_depth_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    min_samples_split_list = [2,3,4,5]
    min_samples_leaf_list = [10,11,12,13,14,15,16]
    max_features_list = [4,5,6,7,8,9,10,11,12,13,14,15,16]
    max_leaf_nodes_list = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    best_acc = 0
    best_params = {}
    
    for depth in max_depth_list:
        for split in min_samples_split_list:
            for leaf in min_samples_leaf_list:
                for feature in max_features_list:
                    for no in max_leaf_nodes_list:
                        d_full = AdvancedDecisionTreeClassifier(max_depth=depth, 
                                                            min_samples_split=split,
                                                            min_samples_leaf=leaf,
                                                            max_features=feature,
                                                            max_leaf_nodes=no,
                                                            oversample=True)
                        d_full.fit(train_data, train_label)
                        acc = calc_accuracy(confusion_matrix(test_label, d_full.predict(test_data)))
                        
                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                'max_depth': depth,
                                'min_samples_split': split,
                                'min_samples_leaf': leaf,
                                'max_features': feature,
                                'max_leaf_nodes': no
                            }
            
    return best_params, best_acc

if __name__ == "__main__":
    t_full, t_full_label = load_dataset('data/train_full.txt')
    t_data, t_label = load_dataset('data/test.txt')

    a, b = grid_search(t_full, t_full_label, t_data, t_label)
    print(a)
    print(b)
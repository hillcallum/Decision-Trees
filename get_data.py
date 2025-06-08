import numpy as np
import os
import matplotlib.pyplot as plt

def load_dataset(filepath):
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found at {filepath}")

    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    training_data = []
    training_label = []

    for line in lines:
        if line.strip() != " ":
            parts = line.strip().split(',')  
            training_data.append(list(map(int, parts[:-1])))
            training_label.append(parts[-1])

    return np.array(training_data), np.array(training_label)

def unique_labels(data):
    return np.unique(data)

def count_label_proportions(data):
    unique_vals, counts = np.unique(data, return_counts=True)
    proportions = np.round(counts / len(data), 2)
    entropy = np.round(-sum(p * np.log2(p) for p in proportions if p > 0), 3)
    return dict(zip(unique_vals, proportions)), entropy

def number_of_samples(data):
    return len(data)

def data_information(xtrain, xlabel):
    print("-"*90)
    print("Dimensions of training data (K, N) - (",len(xtrain[0]),",", len(xtrain),")")
    print("Dimensions of label data (N, ) - (",len(xlabel),", )")
    print("-"*90)
    print("Number of unique labels: ", unique_labels(xlabel))
    print("Number of samples: ", number_of_samples(xlabel))
    proportions, entropy = count_label_proportions(xlabel)
    print("Proportions of labels: ", proportions)
    print("Entropy of labels: ", entropy)
    print("-"*90, "\n")


def plot_label_distribution(filepaths, labels):
    
    distributions = []  
    
    for filepath in filepaths:
        _, y_labels = load_dataset(filepath)
        label_counts = {label: 0 for label in labels} 

       
        for label in y_labels:
            label_counts[label] += 1
        
        total_samples = len(y_labels)
        proportions = [label_counts[label] / total_samples for label in labels]
        distributions.append(proportions) 
        

   
    distributions = np.array(distributions)

    
    x = np.arange(len(labels))  
    bar_width = 0.25  
    fig, ax = plt.subplots(figsize=(10, 6))

 
    for i, filepath in enumerate(filepaths):
        ax.bar(
            x + i * bar_width,
            distributions[i],
            width=bar_width,
            label=f"{os.path.basename(filepath)}"
        )

    ax.set_xticks(x + bar_width * (len(filepaths) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Class Labels")
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Class Labels per File")
    ax.legend()
    plt.tight_layout()
    plt.show()


def attribute_stats(filepath):
 
    xtrain, _ = load_dataset(filepath)  
    filepath = os.path.basename(filepath)
    print(f"Statistics for file: {filepath}")
    print("-" * 90)
    print(f"{'Attribute':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Median':<10} {'Std':<10}")
    print("-" * 90)

    for i in range(xtrain.shape[1]):
        attribute = xtrain[:, i]  
        min_val = np.min(attribute)
        max_val = np.max(attribute)
        mean_val = np.mean(attribute)
        median_val = np.median(attribute)
        std_val = np.std(attribute)
        print(f"{i+1:<10} {min_val:<10.2f} {max_val:<10.2f} {mean_val:<10.2f} {median_val:<10.2f} {std_val:<10.2f}")
    print("-" * 90, "\n")

def compare_datasets(file1, file2):
    file1_name = os.path.basename(file1)
    file2_name = os.path.basename(file2)
    _, labels1 = load_dataset(file1)
    _, labels2 = load_dataset(file2)
   
    diff_count = np.sum(labels1 != labels2)
    proportion_diff = diff_count / len(labels1)
    print(f"\nComparing {file1_name} with {file2_name}")
    print(f"Proportion of differing labels: {proportion_diff:.2%}\n")

    dist1, _ = count_label_proportions(labels1)
    dist2, _ = count_label_proportions(labels2)
    
    print(f"{'Class':<10} {file1_name:<15} {file2_name:<15} {'Difference':<10}")
    print("-" * 50)

    for label in set(dist1.keys()).union(dist2.keys()):
        count1 = dist1.get(label, 0)  
        count2 = dist2.get(label, 0)  
        diff = count1 - count2       
        print(f"{label:<10} {count1:<15.2%} {count2:<15.2%} {diff:<10.2%}")

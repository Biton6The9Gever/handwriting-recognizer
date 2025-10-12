import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random


def plot_label_distribution(labels):
    """Plot the frequency of each label."""
    plt.figure(figsize=(10, 4))
    sns.countplot(x=labels)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_random_samples(X, y, num_samples=9):
    """Display random sample images."""
    labels = X.flatten()
    n = int(np.ceil(np.sqrt(num_samples)))

    plt.figure(figsize=(8, 8))
    indices = random.sample(range(len(y)), num_samples)
    for i, idx in enumerate(indices):
        plt.subplot(n, n, i + 1)
        plt.imshow(y[idx])
        plt.title(str(labels[idx]))
        plt.axis('off')

    plt.suptitle("Random Sample Images", fontsize=14)
    plt.tight_layout()
    plt.show()



# main wrapper 

def visualize_all(X, y, num_samples=9):
    """
    Run all visualization functions on the dataset.
    """
    print("[INFO] Starting full dataset visualization...")
    labels = X.flatten()

    plot_label_distribution(labels)
    show_random_samples(X, y, num_samples=num_samples)

    print("[DONE] All visualizations complete.")

import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random


"""
Data Preprocessing utils function below
"""

def random_split(X, y, avg_num_per_class):
    num_classes = np.max(y) + 1
    num_samples = avg_num_per_class * num_classes
    
    idx1 = np.random.choice(np.arange(len(y)), size=num_samples, replace=False)
    idx2 = ~np.isin(np.arange(len(y)), idx1) 
    X1, y1 = X[idx1], y[idx1]
    X2, y2 = X[idx2], y[idx2]
    
    return X1, y1, X2, y2


def random_split_jax(X, y, avg_num_per_class, key=None):
    if key is None:
        key = random.PRNGKey(0)

    num_classes = jnp.max(y) + 1
    num_samples = avg_num_per_class * num_classes
    indices = jnp.arange(len(y))

    shuffled_indices = random.permutation(key, indices)
    idx1 = shuffled_indices[:num_samples]
    idx2 = shuffled_indices[num_samples:]
    X1, y1 = X[idx1], y[idx1]
    X2, y2 = X[idx2], y[idx2]

    return X1, y1, X2, y2


def reinitClasses(labels, rare_classes):
    """
    Function for remapping after filtering out rare classes
    """
    # Identify non-rare samples
    remaining_idx = ~np.isin(labels, rare_classes)
    
    # Extract non-rare labels
    remaining_labels = labels[remaining_idx]
    
    # Use np.unique to get unique labels and remap labels
    unique_labels, remapped_labels = np.unique(remaining_labels, return_inverse=True)
    
    # Create remapping dictionary
    remapping = {original_label: new_label for new_label, original_label in enumerate(unique_labels)}
    
    return remaining_idx, remapped_labels, remapping

"""
Data loading
"""
# Load CIFAR-100 Data from .npy files
def load_cifar100_data(scores_file='data/results_scores.npy', labels_file='data/results_labels.npy'):
    # Get the absolute path to the data files
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scores_path = os.path.join(base_dir, scores_file)
    labels_path = os.path.join(base_dir, labels_file)

    softmax_scores = jnp.array(jnp.load(scores_path))
    labels = jnp.array(jnp.load(labels_path))
    return softmax_scores, labels
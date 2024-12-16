import numpy as np
import jax.numpy as jnp

def computeClassCoverage(actual_labels, predicted_sets):
    total_classes = max(actual_labels) + 1
    coverage_per_class = np.zeros(total_classes)
    for class_label in range(total_classes):
        class_indices = np.where(actual_labels == class_label)[0]
        class_predicted_sets = (predicted_sets[i] for i in class_indices)
        coverage_per_class[class_label] = sum(1 for prediction_set in class_predicted_sets if class_label in prediction_set) / len(class_indices)
    return coverage_per_class

def compute_coverage_metrics(true_labels, prediction_sets, alpha):

    num_samples = len(true_labels)
    assert num_samples == len(prediction_sets), "Number of labels and prediction sets must match."

    # Determine if the true label is in the prediction set for each sample
    hits = [true_labels[i] in prediction_sets[i] for i in range(num_samples)]

    # Compute overall coverage
    coverage = np.mean(hits)

    # Prepare results
    coverage_results = {
        'coverage': coverage,
        'covGap': coverage - (1 - alpha),
    }

    return coverage_results


def compute_set_size_metrics(prediction_sets):
    
    set_sizes = [len(pred_set) for pred_set in prediction_sets]

    size_metrics = {
        'mean_size': np.mean(set_sizes),
        'std_size': np.std(set_sizes),
    }

    return size_metrics


def compute_all_metrics(val_labels, preds, alpha, cluster_assignments=None):
    class_cond_cov = computeClassCoverage(val_labels, preds)
        
    # Average class coverage gap
    avg_class_cov_gap = np.mean(np.abs(class_cond_cov - (1-alpha)))

    # Average class coverage gap error
    avg_class_cov_gap_std = np.std(np.abs(class_cond_cov - (1-alpha)))

    class_cov_metrics = {'mean_class_cov_gap': avg_class_cov_gap, 
                         'cov_gap_std': avg_class_cov_gap_std,
                        }

    curr_set_sizes = [len(x) for x in preds]
    set_size_metrics = {'set_size_mean': np.mean(curr_set_sizes), 
                        'set_size_std': np.std(curr_set_sizes),
                        '[.25, .5, .75, .9] quantiles': np.quantile(curr_set_sizes, [.25, .5, .75, .9])
                        }
    
    return class_cov_metrics, set_size_metrics


def computeAPS_scores(softmax_scores):
    # Step 1: Sort the softmax scores in descending order for each sample
    sorted_indices = jnp.argsort(-softmax_scores, axis=1)
    sorted_softmax = jnp.take_along_axis(softmax_scores, sorted_indices, axis=1)

    # Step 2: Compute cumulative sums of the sorted softmax scores
    cumulative_probs = jnp.cumsum(sorted_softmax, axis=1)

    # Step 3: Map the cumulative sums back to the original class order
    inv_sorted_indices = jnp.argsort(sorted_indices, axis=1)
    cumulative_probs_original = jnp.take_along_axis(cumulative_probs, inv_sorted_indices, axis=1)

    # Step 4: Compute the APS conformity scores for all classes
    aps_scores = cumulative_probs_original - softmax_scores

    return aps_scores

def computeRAPS_scores(softmax_scores, lambda_param, k_reg):
    # Step 1: Sort the softmax scores in descending order for each sample
    sorted_indices = jnp.argsort(-softmax_scores, axis=1)
    sorted_softmax = jnp.take_along_axis(softmax_scores, sorted_indices, axis=1)

    # Step 2: Compute cumulative sums of the sorted softmax scores
    cumulative_probs = jnp.cumsum(sorted_softmax, axis=1)

    # Step 3: Map the cumulative sums back to the original class order
    inv_sorted_indices = jnp.argsort(sorted_indices, axis=1)
    cumulative_probs_original = jnp.take_along_axis(cumulative_probs, inv_sorted_indices, axis=1)

    # Step 4: Compute the rank of each class (1-based rank)
    ranks = inv_sorted_indices + 1  # Ranks start from 1

    # Step 5: Compute the regularization term
    reg_term = jnp.maximum(lambda_param * (ranks - k_reg), 0)

    # Step 6: Add the regularization term to the cumulative probabilities
    scores = cumulative_probs_original + reg_term

    # Step 7: Compute RAPS scores
    raps_scores = scores - softmax_scores

    return raps_scores
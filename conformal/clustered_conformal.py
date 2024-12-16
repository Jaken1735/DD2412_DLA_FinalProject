import jax.numpy as jnp
from collections import Counter
import numpy as np
from conformal.standard_conformal import *
from conformal.classwise_conformal import *

def quantileThreshold(significanceLevel):
    """
    Compute the smallest sample count threshold such that the quantile estimation satisfies:
    ceil((sampleCount + 1) * (1 - significanceLevel) / sampleCount) <= 1.
    This threshold helps in identifying classes with sufficient samples.
    """
    sampleCount = 1
    # Increment sampleCount until the threshold condition is satisfied
    while jnp.ceil((sampleCount + 1) * (1 - significanceLevel) / sampleCount) > 1:
        sampleCount += 1
    return sampleCount

def rareClasses(classLabels, significanceLevel, numberOfClasses):
    """
    Identify rare classes based on the significance level.
    A class is considered rare if its sample count falls below a computed threshold.
    """
    # Step 1: Compute the minimum required sample count using the quantile threshold
    smallestSampleCount = quantileThreshold(significanceLevel)
    # Step 2: Get unique class labels and their corresponding sample counts
    uniqueClasses, uniqueCounts = jnp.unique(classLabels, return_counts=True)
    # Step 3: Identify classes with counts below the threshold
    rareClasses = uniqueClasses[uniqueCounts < smallestSampleCount]
    # Step 4: Identify classes that have zero samples and combine them with rare classes
    classSet = set(uniqueClasses.tolist())
    zeroSampleClasses = [cls for cls in range(numberOfClasses) if cls not in classSet]
    rareClasses = jnp.concatenate((rareClasses, jnp.array(zeroSampleClasses)))
    
    return rareClasses

def selecting_hparameters(totalCalibrationLabels, numberOfClasses, significanceLevel):
    """
    Select hyperparameters for clustering based on the sample counts and significance level.
    """
    # Step 1: Count occurrences of each class label
    classCountDict = Counter(np.asarray(totalCalibrationLabels))
    classCounts = [classCountDict.get(cls, 0) for cls in range(numberOfClasses)]
    
    # Step 2: Compute the threshold count for identifying rare classes
    thresholdCount = quantileThreshold(significanceLevel)
    
    # Step 3: Ensure the minimum count satisfies the threshold
    minCount = max(min(classCounts), thresholdCount)
    remainingClassCount = jnp.sum(jnp.array(classCounts) >= minCount)
    
    # Step 4: Estimate clustering parameters
    clusteringSampleCount = minCount * remainingClassCount // (75 + remainingClassCount)     # Compute clustering sample count using a heuristic formula
    clusterCount = clusteringSampleCount // 2     # Compute the number of clusters as half the clustering sample count (rounded down)
    
    # Compute the fraction of samples to use for clustering
    clusteringFraction = clusteringSampleCount / minCount
    
    return clusteringSampleCount, clusterCount, clusteringFraction

def clusterSpecificQhats(clusterAssignments, calibrationScoresAll, calibrationTrueLabels, significanceLevel, nullQuantile=True):
    """
    Compute cluster-specific quantiles (q-hats) by aggregating scores based on cluster assignments.
    """
    # Handle default null quantile
    if nullQuantile:
        nullQuantile = computeQhat(calibrationScoresAll, significanceLevel)
    
    # Step 1: Extract scores for the true labels
    if calibrationScoresAll.ndim == 2:  # Multi-class case
        calibrationScoresAll = calibrationScoresAll[np.arange(len(calibrationTrueLabels)), calibrationTrueLabels]
    
    # Step 2: If no clustering was applied
    if np.all(clusterAssignments == -1):
        return np.full(clusterAssignments.shape, nullQuantile)

    # Step 3: Map true labels to their corresponding clusters
    calibrationTrueClusters = clusterAssignments[calibrationTrueLabels]
    maxClusterIndex = np.max(clusterAssignments)
    numberOfClusters = maxClusterIndex + 1 if maxClusterIndex >= 0 else 0
    
    # Compute quantiles for each cluster
    clusterQuantiles = calculate_classwise_q_hat(scores_all=calibrationScoresAll, true_labels=calibrationTrueClusters, n_classes=numberOfClusters, alpha=significanceLevel)

    # Map cluster quantiles back to class indices
    numberOfClasses = len(clusterAssignments)
    classQuantiles = np.full(numberOfClasses, nullQuantile)
    validClusters = clusterAssignments >= 0
    classQuantiles[validClusters] = clusterQuantiles[clusterAssignments[validClusters]]

    return classQuantiles

def computeQhat(sampleScores, significanceLevel):
    """
    Compute the quantile q-hat for standard conformal prediction.
    """
    sampleCount = len(sampleScores)
    quantileIndex = int(jnp.ceil((sampleCount + 1) * (1 - significanceLevel)))
    quantileIndex = min(max(quantileIndex, 1), sampleCount)
    sortedScores = jnp.sort(sampleScores)
    quantileValue = sortedScores[quantileIndex - 1]
    return quantileValue



""""
Fetched from original paper
"""

def quantile_embedding(samples, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    Computes the q-quantiles of samples and returns the vector of quantiles
    '''
    return np.quantile(samples, q)

def embed_all_classes(scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
    '''
    Input:
        - scores_all: num_instances x num_classes array where 
            scores_all[i,j] = score of class j for instance i
          Alternatively, num_instances-length array where scores_all[i] = score of true class for instance i
        - labels: num_instances-length array of true class labels
        - q: quantiles to include in embedding
        - return_cts: if True, return an array containing the counts for each class 
        
    Output: 
        - embeddings: num_classes x len(q) array where ith row is the embeddings of class i
        - (Optional) cts: num_classes-length array where cts[i] = # of times class i 
        appears in labels 
    '''
    num_classes = len(np.unique(labels))
    
    embeddings = np.zeros((num_classes, len(q)))
    cts = np.zeros((num_classes,))
    
    for i in range(num_classes):
        if len(scores_all.shape) == 2:
            class_i_scores = scores_all[labels==i,i]
        else:
            class_i_scores = scores_all[labels==i] 
        cts[i] = class_i_scores.shape[0]
        embeddings[i,:] = quantile_embedding(class_i_scores, q=q)
    
    if return_cts:
        return embeddings, cts
    else:
        return embeddings
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import numpy as np
import operator
import random
import math
digits = load_digits()

"""
An implementation of the k-means clustering algorithm for hand-written
digit recognition using Python.
Author: 680039562
Year: 2020
"""

def load_sets(split, training=[], test=[]):
    """
    Loads the values for the training and test
    datasets.
    :param split: ratio of training to test size
    :param training: training dataset
    :param test: test dataset
    """
    for x in range(len(digits.data)):
        if random.random() < split: # Ratio split between training and test
            training.append((digits.target[x],digits.data[x]))
        else:
            test.append((digits.target[x],digits.data[x]))

def rand_centroids(training, k):
    """
    Creates k centroids by choosing k data points
    from the training dataset at random.
    :param training: training data set
    :param k: number of centroids to create
    :return: a list of new centroid values
    """
    random_data_points = []
    random_data_points = random.sample(training, k)
    centroid_list = []
    for x in random_data_points: # Selects k unlabelled centroids at random
        centroid_list.append(x[1])
    return centroid_list

def sum_cluster(cluster):
    """
    Performs element-wise addition on cluster
    vectors.
    :param cluster: a labelled cluster
    :return: the element-wise sum of vector arrays
    """
    sum = [0.0] * len(cluster[0][1]) # Set 64 feature size
    sum = np.array(sum)
    for (target, vector) in cluster:
        sum += vector # Element wise addition of vectors
    return sum

def mean_cluster(cluster):
    """
    Calculates the element-wise mean for
    each cluster vector.
    :param cluster: a labelled cluster
    :return: the element-wise mean of the cluster vector arrays
    """
    ew_sum_cluster  = sum_cluster(cluster) # Element wise sum of cluster
    ew_mean_cluster = ew_sum_cluster / float(len(cluster)) # Mean of cluster
    return ew_mean_cluster

def create_clusters(training, centroid_list, k):
    """
    Creates k clusters containing labelled data points. Each
    cluster has data points close to its certain centroid.
    :param training: labelled training dataset
    :param centroid_list: list of unlabelled centroids
    :param k: number of clusters
    :return: dictionary values of cluster dataset values
    """
    cluster_dict = {}
    cluster_list = [[] for _ in range(k)] # Creates k cluster classes
    for (index, list) in enumerate(cluster_list):
        cluster_dict[index] = list # Create dictionary of index and cluster

    for (target, data_point) in training: # Go through each training point
        minimum = float("inf")
        cluster_index = -1 # What cluster data_point belongs to
        for (cluster_val, centroid) in enumerate(centroid_list):
            distance = np.linalg.norm(data_point - centroid) # Euc distance
            if distance < minimum:
                minimum = distance
                cluster_index = cluster_val # What cluster data belongs to
        cluster_dict[cluster_index].append((target, data_point))
    return cluster_dict.values()

def new_centroid(cluster):
    """
    Re-positions the centroid for each k, cluster.
    :param cluster: cluster of labelled training dataset
    :return: list of newly positioned centroids
    """
    new_centroid_list = []
    for cluster_val in cluster:
        new_centroid_list.append(mean_cluster(cluster_val)) # Centroid value
    return new_centroid_list

def convergence(training, centroid_list, cluster, max_iter, k):
    """
    Calculate the final cluster labelled data points and their
    cluster centroids.
    :param training: the training dataset
    :param centroid_list: the initial centroid list
    :param cluster: the initial labelled cluster
    :param max_iter: max k iterations per single run
    :param k: number of clusters
    :return: final labelled cluster and centroid list
    """
    max_limit = max_iter
    while max_iter != 0:
        old_centroid_list = centroid_list # Copy centroids
        centroid_list = new_centroid(cluster) # Calculate new centroids
        cluster = create_clusters(training, centroid_list, k)

        # Calculate difference between old and new centroid for each index
        # If the difference is negligable, then convergence
        max_difference = float("-inf")
        for (index, old_centroid) in enumerate(old_centroid_list):
            difference = np.linalg.norm(old_centroid-centroid_list[index])
            if difference > max_difference: # Max centroid move distance
                max_difference = difference

        if max_difference == 0.0:
            print("K:", k, "Iterations:", max_limit-max_iter, " Converged.")
            break

        max_iter -= 1
    return cluster, centroid_list

def get_label(clusters, centroids):
    """
    Returns a tuple of labelled centroids.
    :param clusters: labelled clusters dictionary
    :param centroids: centroids of each cluster
    :return: tuple of (target, centroid)
    """
    labelled_centroids = []
    for (cluster_index, cluster) in enumerate(clusters): # For each cluster
        target_count = {}
        for (target, data_point) in cluster: # Each instance
            if target not in target_count:
                target_count[target]  = 1
            else:
                target_count[target] += 1
        # Get max target
        label = max(target_count.items(), key=operator.itemgetter(1))[0]
        labelled_centroids.append((label, centroids[cluster_index]))
    return labelled_centroids

def check_accuracy(labelled_centroids, test):
    """
    Calculate the error percentage using our labelled centroids
    and testing dataset.
    :param labelled_centroids: labelled centroid of a cluster
    :param testing: labelled testing data set
    :return: error percentage
    """
    correct = 0
    for (target, data_point) in test:
        # Find closest straight line centroid
        minimum = float("inf")
        pred_v = -1
        for (label, centroid) in labelled_centroids:
            distance = np.linalg.norm(data_point - centroid)
            if distance < minimum:
                minimum = distance
                pred_v = label
        if pred_v == target:
            correct +=1
    return round(correct/float(len(test)) * 100, 3)

def plot_clusters_centroids(clusters, centroids):
    """
    Plot clusters and centroids for visualisation of
    k-means clustering.
    :param clusters: the labelled clusters
    :param centroids: the centroids
    """
    centroid_plot = np.array(centroids)
    pca = PCA(n_components=2).fit_transform(centroid_plot) # Reduce data set
    plt.scatter(pca[:, 0], pca[:, 1], marker='x', s=100, linewidths=3,
                color='r', zorder=10)

    cluster_plot = []
    for cluster in clusters:
        for (target, data_point) in cluster:
            cluster_plot.append(data_point)

    # Plot clusters vs centroid 2-dimensional graph
    pca = PCA(n_components=2).fit_transform(cluster_plot) # Reduce data set
    plt.scatter(pca[:,0], pca[:,1], s=5, color='black')
    plt.title("K Means Visualised - Red Crosses Centroid Marker")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return

def plot_accuracy_vs_k(training, test, k_upper):
    """
    Plot a graph of accuracy of clustering against
    the varied values of k.
    :param training: unlabelled training set
    :param test: labelled training set
    :param k_range: upper bound of k range
    """
    accuracy_k = {} # Accuracy of each cluster
    for k in range(1, k_upper+1): # Cannot have zero clusters
        init_centroid_list = rand_centroids(training, k)
        label_cluster = create_clusters(training, init_centroid_list, k)
        clusters, centroids = convergence(training, init_centroid_list,
                                          label_cluster, 300, k)
        labelled_centroids = get_label(clusters, centroids)
        acc_rate = check_accuracy(labelled_centroids, test)
        print("Accuracy rate:", acc_rate, "%")
        accuracy_k[k] = acc_rate
    # Plot of sorted items
    data = sorted(accuracy_k.items())
    x, y = zip(*data)
    plt.title("Accuracy against size of clusters")
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.xlabel("Number of clusters")
    plt.ylabel("Accuracy of clustering")
    plt.show()
    return

def elbow_method(training, k_range):
    """
    Elbow method plots the sum of squared
    distances between each data point (wcss)
    and it's centroid versus the k value.
    :param training: training dataset
    :param k_range: range of clusters to plot
    """
    wcss = {index: 0 for index in range(1, k_range+1)}
    for k in range(1, k_range+1): # Cannot have zero clusters
        init_centroid_list = rand_centroids(training, k)
        label_cluster = create_clusters(training, init_centroid_list, k)
        clusters, centroids = convergence(training, init_centroid_list,
                                          label_cluster, 300, k)
        # Enumerate cluster to get centroid
        for (cluster_index, cluster) in enumerate(clusters):
            wcss_sum = 0 # Calculate wcss for each cluster
            for (target, data_point) in cluster:
                wcss_sum += math.pow(np.linalg.norm(data_point-
                                     centroids[cluster_index]),2)
        wcss[k] = wcss_sum
    data = sorted(wcss.items())
    x, y = zip(*data) # Unpack K value and WCSS sum
    plt.title("Elbow Point Graph")
    plt.plot(x, y)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()
    return

def make_cluster_table(clusters):
    """
    Creates a table of cluster labels and the
    frequency of digits in the cluster.
    :param clusters: labelled clusters
    """
    freq_table = PrettyTable(['Cluster Label','0','1','2','3','4','5','6','7',
                             '8','9'])
    for cluster in clusters:
        frequency = {index: 0 for index in range(10)} # Set dictionary value
        for (target, data_point) in cluster:
            frequency[target] += 1 # Update frequency of each label in cluster
        # Row gets the cluster label by getting most popular label in cluster
        row = [max(frequency.items(), key=operator.itemgetter(1))[0]]
        for freq in frequency.values():
            row.append(freq)
        freq_table.add_row(row)
    print(freq_table)
    return

def main():
    """
    Main Function to execute program.
    """
    training = [] # This is the training set
    test = [] # This is the test set to measure the accuracy of our clustering
    split = 2.0/3 # Training to Test ratio
    load_sets(split, training, test)

    k = 10
    max_iter = 300
    centroid_list = rand_centroids(training, k) # K rand centroids
    label_cluster = create_clusters(training, centroid_list, k) # K clusters
    clusters, centroids = convergence(training, centroid_list, label_cluster,
                                      max_iter, k)
    labelled_centroids = get_label(clusters, centroids)
    acc_rate = check_accuracy(labelled_centroids, test)
    print("Accuracy rate:", acc_rate, "%")

    # Create a table of cluster and digit frequency
    print("\nFrequency table, K =", k)
    make_cluster_table(clusters)

    # Plot clusters and centroids visualisation
    print("\nPlotting Visualisation.")
    plot_clusters_centroids(clusters, centroids)

    # Plot the accuracy rate of clustering for varied sizes of 1 to K
    k_limit = 15
    print("\nPlotting...")
    plot_accuracy_vs_k(training, test, k_limit)

    # Create an Elbow Point graph to find optimum K value
    k_range = 30
    print("\nCalculating WCSS...")
    elbow_method(training, k_range)


if __name__ == "__main__":
    main()

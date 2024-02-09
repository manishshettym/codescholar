from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch

from codescholar.utils.search_utils import read_prog, read_graph, read_embeddings_redis


def extract_textual_features(args, prog_indices):
    programs = [read_prog(args, idx) for idx in prog_indices]
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(programs)
    return tfidf_matrix.toarray()


def extract_graph_features(args, prog_indices):
    graph_embs = read_embeddings_redis(args, prog_indices)
    graph_features = [torch.mean(f, axis=0).numpy() for f in graph_embs]
    return np.array(graph_features)


def cluster_programs(args, prog_indices, n_clusters=10):
    textual_features = extract_textual_features(args, prog_indices)
    graph_features = extract_graph_features(args, prog_indices)
    combined_features = np.hstack((textual_features, graph_features))

    scaler = StandardScaler()
    pca = PCA(n_components=0.95, random_state=42)
    standardized_features = scaler.fit_transform(combined_features)
    reduced_features = pca.fit_transform(standardized_features)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reduced_features)
    centroids = kmeans.cluster_centers_

    distances = euclidean_distances(reduced_features, centroids)
    closest_point_indices = np.argmin(distances, axis=0)
    center_indices = [prog_indices[idx] for idx in closest_point_indices]

    print("Number of cluster centers: ", len(center_indices))

    return center_indices

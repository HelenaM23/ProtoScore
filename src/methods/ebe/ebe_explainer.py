""" "
Code from this file is inspired by https://github.com/nesl/ExMatchina
"""

import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import Model, layers

from .utils.utils import ConceptProperties
from .visualization import viszualization


def separate_model(model: Model):
    """
    separate last layer from model and return two separate models
    """
    num_layers = len(model.layers)
    feature_model = Model(
        inputs=model.input, outputs=model.layers[num_layers - 2].output
    )
    pred_model_shape = model.layers[num_layers - 2].output.shape
    pred_model_shape = pred_model_shape[1:]  # Remove Batch from front.
    pred_model_input = layers.Input(shape=pred_model_shape)
    x = pred_model_input
    for layer in model.layers[num_layers - 1 :]:  # idx + 1
        x = layer(x)
    pred_model = Model(pred_model_input, x)
    return feature_model, pred_model


def get_examples(model: Model, train, test, n_concepts, metric="euc_dist"):

    feature_model, pred_model = separate_model(model)
    # here we could embed the training and the test data. Learn the clusters from the train data but then pick the closest sample from the test data
    embeddings = feature_model.predict(train.X)

    kmeans = KMeans(n_clusters=n_concepts, random_state=0).fit(embeddings)
    centers = kmeans.cluster_centers_
    k = 1  # find closest k signals to center
    X_examples = np.zeros((n_concepts * k, len(train.X[0]), len(train.X[0, 0])))
    y_examples = np.zeros((n_concepts * k))
    for i, center in enumerate(centers):
        if metric == "cosine":
            distance = cosine_similarity(embeddings, centers[i : i + 1, :])[:, 0]
            indices = np.argsort(distance)[-k:]
        elif metric == "dot":
            distance = (embeddings @ centers[i : i + 1, :].T)[:, 0]
            indices = np.argsort(distance)[:k]
        elif metric == "euc_dist":
            distance = np.linalg.norm(embeddings - centers[i : i + 1, :], axis=1)
            indices = np.argsort(distance)[:k]
        # TODO here they used originally the test set but they compute the indices for the train set ? we can either pick the samples from the train set or compute the similarity using the test set instead of the train set
        # X_examples[i:i+k] = test.X[indices]
        X_examples[i : i + k] = train.X[indices]

    y_examples = model.predict(X_examples)

    return X_examples, y_examples


def get_ebe_explanations(
    model: tf.keras.Model,
    train: namedtuple,
    test: namedtuple,
    output_dir: Path,
    n_concepts: int,
):
    """
    Function extracts explanation by example from trained model and stores all concepts and their properties into
    output folder
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param n_concepts: number of concepts to extract
    """
    #
    output_dir_exbyex = output_dir / "ebe"
    output_dir_exbyex.mkdir(parents=True, exist_ok=True)
    X_examples, y_examples = get_examples(
        model=model, train=train, test=test, n_concepts=n_concepts
    )
    viszualization.plot_concepts_by_label(
        X_examples,
        output_dir / "example_explanations.png",
        y_examples,
        n_labels=train.y.shape[1],
    )

    # get concept properties
    cp = ConceptProperties()

    feature_model, pred_model = separate_model(model)
    latent = feature_model(test.X)
    # Prototypes
    concepts = feature_model(X_examples)
    with open(output_dir / "prototypes.pkl", "wb") as f:
        pickle.dump(concepts, f)

    print("Embeddings saved to embeddings.pkl.")
    y_pred = model(test.X)
    instance_concept = cp.get_closest_rec_concept_to_instance(
        test.X, latent.numpy(), concepts.numpy()
    )

    pd.DataFrame(
        {
            "model": "EBE",
            "accuracy": cp.get_completness(test.y, y_pred),
            "output_dir": output_dir,
            "n_concepts": n_concepts,
            "concept representability": cp.KL_divergence_performance(
                test.X[:, :, 0], latent
            ),  #
            "reconstructed concept representability": cp.KL_divergence_performance(
                test.X[:, :, 0], instance_concept[:, :, 0]
            ),
        },
        index=[0],
    ).to_csv(output_dir / "completeness_importance_ebe.csv")

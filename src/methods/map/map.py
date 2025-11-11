# insert this functionality into the orchestrator --> sample usage
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


from .classifier import Classifier

from .map_explainer import Explainer
from .utils.utils import ConceptProperties
from .visualization import viszualization

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     print(physical_devices)
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_classifier(
    model,
    train: namedtuple,
    valid: namedtuple,
    test: namedtuple,
    num_classes: int,
    output_dir,
    filepath,
    hp_path: Path,
) -> tf.keras.Model:
    """
    Function trains black box model
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param n_concepts: number of concepts to extract
    :return: classifier
    """
    # Load hyperparameters classifier
    with open(hp_path, "r") as f:
        classifier_hp_dict = yaml.safe_load(f)

    # define and fit classifier
    clf = Classifier(
        num_classes=num_classes,
        model=model,
        input_shape=np.shape(train.X),
        output_directory=output_dir,
        **classifier_hp_dict
    )
    fit_classifier = not Path(output_dir / "best_model.weights.h5").exists()
    if fit_classifier:
        clf.fit_classifier(train, valid)
    clf.model.load_weights(str(output_dir / "best_model.weights.h5"))

    # eval classifier
    results = clf.model.evaluate(x=test.X, y=test.y, return_dict=True)
    pd.DataFrame.from_dict(results, orient="index").T.to_csv(
        output_dir / "results_classification.csv"
    )

    return clf.model


def get_map_explanations(
    model: tf.keras.Model,
    train: namedtuple,
    test: namedtuple,
    output_dir: Path,
    explainer_name: str,
    n_concepts: int,
    epochs: int,
):
    """
    Function trains model-agnostic explainer and stores all concepts and their properties into output folder
    :param model: classifier to explain
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param explainer_name: name of map to load
    :param n_concepts: number of concepts to extract
    :param epochs: number of epochs to train map
    """
    output_dir_ex = output_dir / explainer_name
    output_dir_ex.mkdir(parents=True, exist_ok=True)

    exp = Explainer(
        input_shape=np.shape(train.X[0]),
        output_directory=output_dir_ex,
        n_concepts=n_concepts,
        latent_dim=n_concepts * 5,
        explainer_name=explainer_name,
        epochs=epochs,
        batch_size=32,
    )

    fit_explainer = not Path(output_dir_ex / "map.weights.h5").exists()
    if fit_explainer:
        exp.fit_explainer(classifier=model, X=train.X, explainer_name=explainer_name)
    else:
        if not exp.explainer.built:
            exp.explainer.build((None,) + np.shape(train.X[0]))
        exp.explainer.load_weights(str(output_dir_ex / "map.weights.h5"))

    X_concepts_kmeans, latent_centers = exp.get_concepts_kmeans(train.X)
    concept_labels = model(X_concepts_kmeans)
    latent = exp.explainer.encoder(test.X)
    viszualization.plot_concepts_by_label(
        X_concepts_kmeans,
        output_dir_ex / "map.png",
        concept_labels,
        n_labels=train.y.shape[1],
    )

    # completeness & importance
    y_pred = model(test.X)

    cp = ConceptProperties()
    map_instance_concept = cp.get_closest_rec_concept_to_instance(
        test.X, latent.numpy(), latent_centers
    )

    pd.DataFrame(
        {
            "model": "MAP",
            "accuracy": cp.get_completness(test.y, y_pred),
            "output_dir": output_dir,
            "n_concepts": n_concepts,
            "concept representability": cp.KL_divergence_performance(
                test.X[:, :, 0], latent
            ),
            "reconstructed concept representability": cp.KL_divergence_performance(
                test.X[:, :, 0], map_instance_concept[:, :, 0]
            ),
            "latent_centers": [latent_centers],
        },
        index=[0],
    ).to_csv(output_dir / "completeness_importance_concept_map.csv")

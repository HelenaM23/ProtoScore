import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from .msp_explainer import PrototypeExplainer
from .utils.utils import ConceptProperties
from .visualization import viszualization

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_msp_explanations(
    train: namedtuple,
    test: namedtuple,
    output_dir: Path,
    n_concepts: int,
    epochs: int,
    n_classes: int,
    filepath: Path,
    hp_filepath: str,
):
    """
    Function trains model-specific explainer and stores all concepts and their properties into output folder
    :param train: training data
    :param test: test data
    :param output_dir: path to output results
    :param n_concepts: number of concepts to extract
    :param epochs: number of epochs to train map
    """
    # Load hyperparameters classifier
    with open(filepath / hp_filepath, "r") as classifier_hyperparameters:
        classifier_hp_dict = yaml.safe_load(classifier_hyperparameters)

    output_dir_protoex = output_dir / "msp"
    output_dir_protoex.mkdir(parents=True, exist_ok=True)

    pexp = PrototypeExplainer(
        input_shape=np.shape(train.X[0]),
        output_directory=output_dir_protoex,
        n_concepts=n_concepts,
        latent_dim=n_concepts,
        n_classes=n_classes,
        explainer_name="prototype1",
        epochs=epochs,
        batch_size=classifier_hp_dict["batch_size"],
    )
    fit_explainer = not Path(output_dir_protoex / "msp.weights.h5").exists()
    if fit_explainer:
        pexp.fit_explainer(X=train.X, y=train.y, lr=classifier_hp_dict["lr"])
    else:
        if not pexp.explainer.built:
            pexp.explainer.build((None,) + np.shape(train.X[0]))
        pexp.explainer.load_weights(str(output_dir_protoex / "msp.weights.h5"))

    reconstructed_prototypes = pexp.get_reconstructed_prototypes()
    prototypes = pexp.explainer.predictor.prototypes
    prototypes_labels = pexp.explainer.predictor(prototypes)
    viszualization.plot_concepts_by_label(
        reconstructed_prototypes,
        output_dir / "msp.png",
        prototypes_labels,
        n_labels=n_classes,
    )

    # # completeness & importance
    cp = ConceptProperties()
    latent = pexp.explainer.encoder(test.X)
    y_pred = pexp.explainer.predictor(latent)

    msp_proto = pexp.explainer.predictor.prototypes
    concept_close_to_activation = cp.get_closest_rec_concept_to_instance_msp(
        test.X, latent, msp_proto
    )

    pd.DataFrame(
        {
            "model": "MSP",
            "accuracy": cp.get_completness(test.y, y_pred),
            "n_concepts": n_concepts,
            "output_dir": output_dir,
            "concept representability": cp.KL_divergence_performance(
                test.X[:, :, 0], latent
            ),
            "reconstructed concept representability": cp.KL_divergence_performance(
                test.X[:, :, 0], concept_close_to_activation[:, :, 0]
            ),
            "latent_centers": [prototypes.numpy()],
        },
        index=[0],
    ).to_csv(output_dir / "completeness_importance_msp.csv")

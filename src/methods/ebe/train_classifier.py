# insert this functionality into the orchestrator --> sample usage
import json
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from .classifier import Classifier


def train_classifier_ebe(
    model,
    train: namedtuple,
    valid: namedtuple,
    test: namedtuple,
    num_classes: int,
    output_dir,
    filepath,
    epochs: int,
    hp_filepath: Optional[
        str
    ] = "prototype_methods_time_series/ebe/classifiers/default_hyperparameters.json",
) -> tf.keras.Model:
    """
    Functions that trains the Explanation by Example Model
    """
    # Load hyperparameters classifier
    with open(filepath / hp_filepath, "r") as classifier_hyperparameters:
        classifier_hp_dict = yaml.safe_load(classifier_hyperparameters)

    classifier_hp_dict["epochs"] = epochs
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

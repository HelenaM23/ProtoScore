import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .msp_explainers import prototype1
from .msp_explainers.autoencoder_helpers import list_of_distances


class PrototypeExplainer:
    """
    Model specific prototype class, proposed by https://github.com/OscarcarLi/PrototypeDL
    """

    def __init__(
        self,
        input_shape: tuple,
        output_directory: Path,
        epochs: int,
        batch_size: int,
        explainer_name: str,
        n_concepts: int,
        n_classes: int,
        latent_dim: int,
        predictor=None,
        build=True,
    ):
        """
        Initializes the model with specified settings
        """
        self.input_shape = input_shape
        self.output_directory = output_directory
        self.epochs = epochs
        self.batch_size = batch_size
        self.explainer_name = explainer_name
        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.predictor = predictor
        if build:
            self.explainer = self.build_explainer()

    def build_explainer(self, **kwargs):
        """
        Builds explainer model
        **kwargs: Keyword arguments for tf.keras.Model.compile method
        :return: Tensorflow model for explanation of time series data
        """
        if self.explainer_name == "prototype1":
            explainer = prototype1.AutoEncoder(
                original_dim=self.input_shape,
                latent_dim=self.latent_dim,
                n_concepts=self.n_concepts,
                n_classes=self.n_classes,
                predictor=self.predictor,
            )
        else:
            raise AssertionError("Model name does not exist")

        return explainer

    def fit_explainer(self, X, y, lr):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        ce_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        mse_loss_fn = tf.keras.losses.MeanSquaredError()

        log_dir = (
            self.output_directory / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        summary_writer = tf.summary.create_file_writer(str(log_dir))

        bat_per_epoch = math.floor(len(X) / self.batch_size)

        for epoch in range(self.epochs):
            print("Start of epoch %d" % (epoch,))

            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_ce = 0.0
            epoch_pdl = 0.0
            epoch_r1 = 0.0
            epoch_r2 = 0.0

            for step in range(bat_per_epoch):
                n = step * self.batch_size
                x_batch_train = X[n : n + self.batch_size].astype(np.float32)
                y_batch_train = y[n : n + self.batch_size]

                with tf.GradientTape() as tape:
                    latent = self.explainer.encoder(x_batch_train)
                    x_reconstructed = self.explainer.decoder(latent)
                    y_pred_reconstructed = self.explainer.predictor(latent)

                    # Loss
                    gamma = 0.05
                    r1_reg_new = self.R1_regularization(latent) * gamma
                    r2_reg_new = self.R2_regularization(latent) * gamma
                    pdl_new = self.PDL1() * 1000

                    mse_loss_new = mse_loss_fn(x_batch_train, x_reconstructed) * gamma
                    ce_loss_new = ce_loss_fn(y_batch_train, y_pred_reconstructed)
                    loss = (
                        mse_loss_new + ce_loss_new + r1_reg_new + r2_reg_new + pdl_new
                    )

                grads = tape.gradient(loss, self.explainer.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.explainer.trainable_weights))

                epoch_loss += loss.numpy()
                epoch_mse += mse_loss_new.numpy()
                epoch_ce += ce_loss_new.numpy()
                epoch_pdl += pdl_new.numpy()
                epoch_r1 += r1_reg_new.numpy()
                epoch_r2 += r2_reg_new.numpy()

            # average across steps
            epoch_loss /= bat_per_epoch
            epoch_mse /= bat_per_epoch
            epoch_ce /= bat_per_epoch
            epoch_pdl /= bat_per_epoch
            epoch_r1 /= bat_per_epoch
            epoch_r2 /= bat_per_epoch

            accuracy = self.get_accuracy(X, y)

            # -- Log summary
            with summary_writer.as_default():
                tf.summary.scalar("loss", epoch_loss, step=epoch)
                tf.summary.scalar("mse_loss", epoch_mse, step=epoch)
                tf.summary.scalar("ce_loss", epoch_ce, step=epoch)
                tf.summary.scalar("pdl", epoch_pdl, step=epoch)
                tf.summary.scalar("r1_reg", epoch_r1, step=epoch)
                tf.summary.scalar("r2_reg", epoch_r2, step=epoch)
                tf.summary.scalar("accuracy", accuracy, step=epoch)

            # -- Print a dictionary of epoch stats
            loss_dict = {
                "epoch": epoch,
                "loss": epoch_loss,
                "mse_loss": epoch_mse,
                "ce_loss": epoch_ce,
                "pdl": epoch_pdl,
                "r1_reg": epoch_r1,
                "r2_reg": epoch_r2,
                "accuracy": accuracy,
            }
            print(loss_dict)

        # Save final metrics and weights
        pd.DataFrame(loss_dict, index=[0]).to_csv(
            str(self.output_directory / "pexp_loss.csv")
        )
        print(self.explainer.predictor.prototypes.numpy())
        if not self.explainer.built:
            self.explainer.build((None,) + self.input_shape)
        self.explainer.save_weights(str(self.output_directory / "pexp_weights.weights.h5"))

    def R1_regularization(self, feature_vectors):
        """
        :param feature_vectors: latent activations of encoder
        :return: R1_regularization
        """
        p = self.explainer.predictor.prototypes
        feature_vector_distances = list_of_distances(p, feature_vectors)
        feature_vector_distances = tf.identity(
            feature_vector_distances, name="feature_vector_distances"
        )
        return tf.reduce_mean(
            tf.reduce_min(feature_vector_distances, axis=1), name="error_1"
        )

    def R2_regularization(self, feature_vectors):
        """
        :param feature_vectors: latent activations of encoder
        :return: R2_regularization
        """
        p = self.explainer.predictor.prototypes
        prototype_distances = list_of_distances(feature_vectors, p)
        prototype_distances = tf.identity(
            prototype_distances, name="prototype_distances"
        )

        return tf.reduce_mean(
            tf.reduce_min(prototype_distances, axis=1), name="error_2"
        )

    def PDL1(self):
        """
        function returns prototype diversity loss according to https://arxiv.org/pdf/1904.08935.pdf
        :return: PDL loss
        """
        p = self.explainer.predictor.prototypes

        distance = []
        for j in range(tf.shape(p)[0] - 1):
            norm = tf.norm(p[j + 1 :] - p[j], axis=-1) ** 2
            min_norm = tf.reduce_min(norm).numpy()
            distance.append(min_norm)
        mean_dist = tf.reduce_mean(distance)
        pdl = 1 / (tf.math.log(mean_dist) + 1e-9)
        return pdl

    def get_accuracy(self, X, y):
        """
        function returns accuracy of learned prototypes
        :return: accuracy of learned prototypes
        """
        latent = self.explainer.encoder(X)
        y_pred = self.explainer.predictor(latent)
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y)
        return accuracy

    def get_reconstructed_prototypes(self):
        """
        function returns learned prototypes
        :return: reconstructed prototypes
        """
        p = self.explainer.predictor.prototypes
        reconstructed_prototypes = self.explainer.decoder(p)
        return reconstructed_prototypes

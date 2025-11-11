import tensorflow as tf
from tensorflow.keras import Model, layers

from ...core.model import CNNBlock


def build_fcn_block_functional(input_shape: tuple, num_classes: int = 2) -> Model:
    """
    Build a Keras Functional model replicating the FCNBlock architecture:
      - 3 CNNBlocks (Conv1D+BN+ReLU)
      - GlobalAveragePooling1D
      - Dense with softmax
    This model can then load weights that were originally trained using the same architecture.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # 1) First CNNBlock
    x = CNNBlock(filters=128, kernel_size=8)(inputs)
    # 2) Second CNNBlock
    x = CNNBlock(filters=256, kernel_size=5)(x)
    # 3) Third CNNBlock
    x = CNNBlock(filters=128, kernel_size=3)(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense output
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Return a Functional model
    model = Model(inputs=inputs, outputs=outputs, name="FCNBlock_Functional")
    return model

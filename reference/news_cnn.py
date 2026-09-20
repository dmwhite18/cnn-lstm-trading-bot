"""Multi-kernel 1D CNN for news-text price-direction classification.

Clean reference implementation of the *news model* from the CNN+LSTM trading bot.
It reads a tokenized news article and outputs P(the stock rises) as a single
sigmoid unit. Faithful to the original 2022 architecture — three parallel Conv1D
branches over a shared embedding, concatenated into a dense head — rewritten for
clarity and modern ``tf.keras``.

Run ``python news_cnn.py`` to print the model summary.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_news_cnn(
    vocab_size: int = 5000,
    max_len: int = 244,
    embed_dim: int = 64,
    conv_filters: int = 256,
    kernel_sizes: tuple[int, ...] = (2, 3, 4),
    dense_units: int = 256,
    embed_dropout: float = 0.2,
    dense_dropout: float = 0.2,
) -> keras.Model:
    """Build and compile the multi-kernel news CNN.

    Args:
        vocab_size: tokenizer vocabulary size (the ``Embedding`` input dim).
        max_len: padded length of each tokenized article.
        embed_dim: embedding vector size.
        conv_filters: number of filters in each parallel Conv1D branch.
        kernel_sizes: kernel width of each branch. Different widths let the
            network pick up 2-, 3- and 4-gram patterns from the same embedding.
        dense_units: width of the first dense head layer (the second is a quarter
            of this).
        embed_dropout: spatial dropout applied to the embedding.
        dense_dropout: dropout applied within the dense head.

    Returns:
        A compiled binary classifier: sigmoid output, ``binary_crossentropy``
        loss. The training label is whether price actually rose over a forward
        window, so the output is an outcome-grounded "will it go up?" probability.
    """
    inp = keras.Input(shape=(max_len,), dtype="int32", name="tokens")
    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(inp)
    x = layers.SpatialDropout1D(embed_dropout, name="embed_dropout")(x)

    # Three parallel convolutional streams over the shared embedding.
    branches = []
    for k in kernel_sizes:
        c = layers.Conv1D(conv_filters, k, activation="relu", name=f"conv_k{k}")(x)
        c = layers.GlobalMaxPooling1D(name=f"pool_k{k}")(c)
        branches.append(c)
    x = layers.concatenate(branches, name="concat")

    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dense_dropout, name="drop_1")(x)
    x = layers.Dense(dense_units // 4, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dense_dropout, name="drop_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_up")(x)

    model = keras.Model(inp, out, name="news_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_news_cnn().summary()

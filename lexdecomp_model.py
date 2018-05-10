import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Lambda, Conv2D
from keras.layers import MaxPooling2D, Flatten, Concatenate, Dense
from keras.layers import Activation, BatchNormalization, Dropout


def semantic_match(X, Y, A, window):
    """Computing semantic match in direction X -> Y
    shape X: (s,n,d), Y: (s,m,d), A: (s, n, m)
    """
    # shape Pivot, lower_lim, upper_lim: (s,n,1)
    Pivot = np.expand_dims(np.argmax(A, axis=-1), axis=-1)
    lower_lim = np.maximum(0, Pivot-window)
    upper_lim = np.minimum(A.shape[-1], Pivot+window)

    # shape indices: (s,n,m)
    # indices = np.tile(np.arange(A.shape[2]), (A.shape[0], A.shape[1] ,1))
    indices = np.tile(np.arange(A.shape[-1]), A.shape[:-1]+(1,))
    # NOTE: To replicate "mcrisc" implementation in github use: indices < upper_lim
    mask = ((indices >= lower_lim) & (indices <= upper_lim)).astype(np.float32)

    # shape X_hat: (n,d)
    X_hat = np.matmul(A*mask, Y)

    return X_hat

def decompose(X, X_hat, method="linear"):
    """Decompose a dataset with regards to its
    semantic match version

    shape X, X_hat: (s,n,d)
    """
    assert method in ("linear", "orthogonal")
    if method == "linear":
        # shape alpha: (s,n,1)
        denom = (np.linalg.norm(X, axis=-1, keepdims=True) *
                 np.linalg.norm(X_hat, axis=-1, keepdims=True))
        alpha = np.divide(np.sum(X * X_hat, axis=-1, keepdims=True),
                          denom, where=denom!=0)

        # shape X_pos, X_neg: (s,n,d)
        X_pos = alpha * X
        X_neg = (1 - alpha) * X
    elif method == "orthogonal":
        # shape X_pos, X_neg: (s,n,d)
        denom = np.sum(X_hat * X_hat, axis=-1, keepdims=True)
        X_pos = np.divide(np.sum(X * X_hat, axis=-1, keepdims=True),
                          denom, where=denom!=0) * X_hat
        X_neg = X - X_pos
    X_pos = np.expand_dims(X_pos, axis=-1)
    X_neg = np.expand_dims(X_neg, axis=-1)
    # shape X_decomp: (s,n,d,2)
    X_decomp = np.concatenate([X_pos, X_neg], axis=-1)
    return X_decomp


def decompose_data(X, Y, window=3, method="linear"):
    """Decompose datasets X, Y into positive and negative
    channels with regards to each other
    shape X: (s,n,d), Y: (s,m,d)
    """
    # Cosine similarity
    # shape A: (s,n,m)
    norm_X = np.linalg.norm(X, axis=-1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=-1, keepdims=True)
    A = np.matmul(np.divide(X, norm_X, where=norm_X!=0), np.swapaxes(np.divide(Y, norm_Y, where=norm_Y!=0), -1, -2))
    A = np.matmul(np.divide(X, norm_X, where=norm_X!=0), np.swapaxes(np.divide(Y, norm_Y, where=norm_Y!=0), -1, -2))

    # Semantic matching
    # shape X_hat: (s,n,d), Y_hat: (s,m,d)
    X_hat = semantic_match(X, Y, A, window=window)
    Y_hat = semantic_match(Y, X, np.swapaxes(A, -1, -2), window=window)
    # Decomposition (pos, neg)
    X_decomp = decompose(X, X_hat, method=method)
    Y_decomp = decompose(Y, Y_hat, method=method)

    return X_decomp, Y_decomp


def transform_data(X, embedding_matrix):
    X_emb = np.zeros(X.shape+(embedding_matrix.shape[1],))
    for i, val in np.ndenumerate(X):
        X_emb[i] = embedding_matrix[val]
    return X_emb


def CNN_encoder(input_shape, embeddings_dim, max_seq_length, filters):
    X_input = Input(input_shape)
    # Applying different filter sizes at the same time
    conv_list = []
    for i, (filter_size, number_of_filters) in enumerate(filters):
        # Convolutional layer
        # Output shape: (batch_size, width_conv, number_of_filters)
        conv = Conv2D(filters=number_of_filters,
                      kernel_size=(filter_size, embeddings_dim),
                      strides=1,
                      padding="valid",
                      data_format="channels_last",
                      name="conv"+str(i))(X_input)
        #conv = BatchNormalization()(conv)
        conv = Activation("tanh")(conv)

        # Max-pooling layer
        # Output shape: (batch_size, 1, number_of_filters)
        width_conv = max_seq_length - filter_size + 1
        conv = MaxPooling2D(pool_size=(width_conv, 1),
                            name="maxpool"+str(i))(conv)
        # Flattening because we only have one layer of conv filters
        # Output shape: (batch_size, number_of_filters)
        conv = Flatten()(conv)

        # storing all conv filters
        conv_list.append(conv)

    # Concatenating the outputs of different filter sizes
    if len(filters) > 1:
        X = Concatenate()(conv_list)
    else:
        X = conv_list[0]

    model = Model(inputs=X_input, outputs=X)
    return model


def lexdecomp_model(input_shape, embeddings_dim, max_seq_length, filters, dropout=0.5, model_type="other"):
    S_input = Input(input_shape)
    T_input = Input(input_shape)

    # Weight-sharing encoder (Siamese architecture)
    if model_type == "siamese":
        encoder = CNN_encoder(input_shape, embeddings_dim, max_seq_length, filters)
        S_encoded = encoder(S_input)
        T_encoded = encoder(T_input)
    else:
        S_encoded = CNN_encoder(input_shape, embeddings_dim, max_seq_length, filters)(S_input)
        T_encoded = CNN_encoder(input_shape, embeddings_dim, max_seq_length, filters)(T_input)

    X = Concatenate()([S_encoded, T_encoded])
    X = Dropout(dropout)(X)
    X = Dense(1, activation="sigmoid")(X)

    model = Model(inputs=[S_input, T_input], outputs=X, name="lexdecomp_model")
    return model

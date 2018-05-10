from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization


class GatedRelevanceNetwork(Layer):
    def __init__(self, output_dim,
            weights_initializer="glorot_uniform",
            bias_initializer="zeros", **kwargs):
        self.output_dim = output_dim
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        super(GatedRelevanceNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size, len1, emb_dim = input_shape[0]
        _, len2, _ = input_shape[1]
        # Weights initialization
        # Bilinear Tensor Product weights
        self.Wb = self.add_weight(name='weights_btp',
                                  shape=(self.output_dim, emb_dim, emb_dim),
                                  initializer=self.weights_initializer,
                                  trainable=True)

        # Single Layer Network weights
        self.Wd = self.add_weight(name='weights_sln',
                                  shape=(2*emb_dim, self.output_dim),
                                  initializer=self.weights_initializer,
                                  trainable=True)

        # Gate weights
        self.Wg = self.add_weight(name='weights_gate',
                                  shape=(2*emb_dim, self.output_dim),
                                  initializer=self.weights_initializer,
                                  trainable=True)

        # Gate bias
        self.bg = self.add_weight(name='bias_gate',
                                  shape=(self.output_dim,),
                                  initializer=self.bias_initializer,
                                  trainable=True)

        # General bias
        self.b = self.add_weight(name='bias',
                                 shape=(self.output_dim,),
                                 initializer=self.bias_initializer,
                                 trainable=True)

        # Channel weights
        self.u = self.add_weight(name="channel_weights",
                                 shape=(self.output_dim, 1),
                                 initializer=self.weights_initializer,
                                 trainable=True)

        super(GatedRelevanceNetwork, self).build(input_shape)

    def call(self, x):
        e1 = x[0]
        e2 = x[1]

        batch_size = K.shape(e1)[0]
        # Usually len1 = len2 = max_seq_length
        _, len1, emb_dim = K.int_shape(e1)
        _, len2, _ = K.int_shape(e2)

        # Repeating the matrices to generate all the combinations
        ne1 = K.reshape(K.repeat_elements(K.expand_dims(e1, axis=2), len2, axis=2),
                       (batch_size, len1*len2, emb_dim))
        ne2 = K.reshape(K.repeat_elements(K.expand_dims(e2, axis=1), len1, axis=1),
                       (batch_size, len1*len2, emb_dim))

        # Repeating the second matrix to use in Bilinear Tensor Product
        ne2_k = K.repeat_elements(K.expand_dims(ne2, axis=-1), self.output_dim, axis=-1)

        # Bilinear tensor product
        btp = K.sum(ne2_k * K.permute_dimensions(K.dot(ne1, self.Wb), (0,1,3,2)), axis=2)
        btp = K.reshape(btp, (batch_size, len1, len2, self.output_dim))

        # Concatenating inputs to apply Single Layer Network
        e = K.concatenate([ne1, ne2], axis=-1)

        # Single Layer Network
        #sln = K.relu(K.dot(e, self.Wd))
        sln = K.tanh(K.dot(e, self.Wd))
        sln = K.reshape(sln, (batch_size, len1, len2, self.output_dim))

        # Gate
        g = K.sigmoid(K.dot(e, self.Wg) + self.bg)
        g = K.reshape(g, (batch_size, len1, len2, self.output_dim))

        # Gated Relevance Network
        #s = K.reshape(K.dot(g*btp + (1-g)*sln + self.b, self.u), (batch_size, len1, len2))
        s = K.dot(g*btp + (1-g)*sln + self.b, self.u)

        return s

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        return (shape1[0], shape1[1], shape2[1], 1)


def BiLSTM_encoder(input_shape, embeddings_dim, embeddings_matrix, word_to_index,
                   max_seq_length, trainable_embeddings, lstm_hidden_units):
    """input shape: (None, max_seq_length)
       embedding_layer shape: (None, max_seq_length, embeddings_dim)
       None refers to the minibatch size."""
    X_input = Input(input_shape)

    # Output shape: (batch_size, max_seq_length, embeddings_dim)
    X = Embedding(len(word_to_index)+1,
                  embeddings_dim,
                  weights=[embeddings_matrix],
                  input_length=None,#max_seq_length,
                  trainable=trainable_embeddings)(X_input)

    # Output shape: (batch_size, max_seq_length, lstm_hidden_units)
    X = Bidirectional(LSTM(lstm_hidden_units, return_sequences=True),
                      merge_mode='concat')(X)

    model = Model(inputs=[X_input], outputs=[X], name="BiLSTM_encoder")
    return model

def gated_relevance_model(input_shape, embeddings_dim, embeddings_matrix,
                          word_to_index, max_seq_length, trainable_embeddings,
                          dropout, lstm_hidden_units, attention_channels,
                          pool_size, fc_hidden_units):
    # TODO: Add docstring
    encoder = BiLSTM_encoder(input_shape, embeddings_dim, embeddings_matrix,
                             word_to_index, max_seq_length, trainable_embeddings,
                             lstm_hidden_units)

    X1_input = Input(input_shape, name="input_X1")
    X2_input = Input(input_shape, name="input_X2")

    # Encoding the inputs using the same weights
    # Output shape: (batch_size, max_seq_length, lstm_hidden_units)
    X1_encoded = encoder(X1_input)
    X2_encoded = encoder(X2_input)

    # Attention matrix
    # Output shape: (batch_size, max_seq_length, max_seq_length, 1)
    X = GatedRelevanceNetwork(attention_channels, name="grn")([X1_encoded, X2_encoded])

    # Non-overlapping 2D max pooling
    # Output shape: (batch_size, pooled_rows, pooled_cols, 1)
    print("shape before pool", X.shape)
    X = MaxPooling2D(pool_size=(pool_size, pool_size),
                     strides=(pool_size, pool_size),
                     padding='valid',
                     data_format="channels_last",
                     name="max_pool")(X)
    X = Flatten()(X)

    # Multi-Layer Perceptron
    #X = Dropout(dropout)(X)
    X = Dense(fc_hidden_units, activation="relu", name="mlp")(X)
    X = Dense(2, activation="softmax", name="output")(X)

    model = Model(inputs=[X1_input, X2_input], outputs=X, name="GRN_model")

    return model

#encoder = Bidirectional(LSTM(lstm_hidden_units, return_sequences=True))

def gated_relevance_model2(input_shape, embeddings_dim, embeddings_matrix,
                          word_to_index, max_seq_length, trainable_embeddings,
                          dropout, lstm_hidden_units, attention_channels,
                          pool_size, fc_hidden_units):
    # TODO: Add docstring
    X1_input = Input(input_shape, name="input_X1")
    X2_input = Input(input_shape, name="input_X2")

    # Encoding the inputs using the same weights
    # Output shape: (batch_size, max_seq_length, lstm_hidden_units)
    X1 = Embedding(len(word_to_index)+1,
                   embeddings_dim,
                   weights=[embeddings_matrix],
                   input_length=max_seq_length,
                   trainable=trainable_embeddings)(X1_input)
    X2 = Embedding(len(word_to_index)+1,
                   embeddings_dim,
                   weights=[embeddings_matrix],
                   input_length=max_seq_length,
                   trainable=trainable_embeddings)(X2_input)

    # Output shape: (batch_size, max_seq_length, lstm_hidden_units)
    X1_encoded = Bidirectional(LSTM(lstm_hidden_units, return_sequences=True))(X1)
    X2_encoded = Bidirectional(LSTM(lstm_hidden_units, return_sequences=True))(X2)

    # Attention matrix
    # Output shape: (batch_size, max_seq_length, max_seq_length, 1)
    X = GatedRelevanceNetwork(attention_channels, name="grn")([X1_encoded, X2_encoded])

    # Non-overlapping 2D max pooling
    # Output shape: (batch_size, pooled_rows, pooled_cols, 1)
    print("shape before pool", X.shape)
    X = MaxPooling2D(pool_size=(pool_size, pool_size),
                     strides=(pool_size, pool_size),
                     padding='valid',
                     data_format="channels_last",
                     name="max_pool")(X)
    X = Flatten()(X)

    # Multi-Layer Perceptron
    #X = Dropout(dropout)(X)
    X = Dense(fc_hidden_units, activation="tanh", name="mlp")(X)
    X = Dense(2, activation="softmax", name="output")(X)

    model = Model(inputs=[X1_input, X2_input], outputs=X, name="GRN_model")

    return model


def gated_relevance_model3(input_shape, embeddings_dim, embeddings_matrix,
                          word_to_index, max_seq_length, trainable_embeddings,
                          dropout, lstm_hidden_units, attention_channels,
                          pool_size, fc_hidden_units):
    # TODO: Add docstring
    X1_input = Input(input_shape, name="input_X1")
    X2_input = Input(input_shape, name="input_X2")

    # Encoding the inputs using the same weights
    # Output shape: (batch_size, max_seq_length, lstm_hidden_units)
    embeddor = Embedding(len(word_to_index)+1,
                   embeddings_dim,
                   weights=[embeddings_matrix],
                   input_length=input_shape[0],
                   trainable=trainable_embeddings,
                   mask_zero=False)
    X1 = embeddor(X1_input)
    X2 = embeddor(X2_input)

    encoder = Bidirectional(LSTM(lstm_hidden_units, return_sequences=True))

    # Output shape: (batch_size, max_seq_length, lstm_hidden_units)
    X1_encoded = encoder(X1)
    X2_encoded = encoder(X2)

    # Attention matrix
    # Output shape: (batch_size, max_seq_length, max_seq_length, 1)
    X = GatedRelevanceNetwork(attention_channels, name="grn")([X1_encoded, X2_encoded])
    #X = BatchNormalization()(X)

    # Non-overlapping 2D max pooling
    # Output shape: (batch_size, pooled_rows, pooled_cols, 1)
    print("shape before pool", X.shape)
    X = MaxPooling2D(pool_size=(pool_size, pool_size),
                     strides=(pool_size, pool_size),
                     padding='valid',
                     data_format="channels_last",
                     name="max_pool")(X)
    X = Flatten()(X)

    # Multi-Layer Perceptron
    #X = Dropout(dropout)(X)
    X = Dense(fc_hidden_units, activation="tanh", name="mlp")(X)
    X = Dropout(dropout)(X)
    X = Dense(2, activation="softmax", name="output")(X)

    model = Model(inputs=[X1_input, X2_input], outputs=X, name="GRN_model")

    return model

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D
from keras.layers import Flatten, Concatenate, Conv2D, Reshape, Activation
from keras.layers import Dense, Lambda, Subtract, Dot, LSTM, Bidirectional
from keras.layers import BatchNormalization
from keras import regularizers
from future_keras import DepthwiseConv2D

def group_a(embedding_layer, filters, poolings, regularizer):
    """ Returns a list with a tensor for each pooling function,
    where each tensor is of the shape:
    [?, len(filters), fnum]"""
    ###########
    # Group A #
    ###########
    outputs = []

    for i, Pool in enumerate(poolings):
        conv_out = []
        for fsize, fnum in filters:
            # if fsize == None do not do convolution
            if fsize:
                X = Conv1D(filters=fnum,
                           kernel_size=fsize,
                           strides=1,
                           padding="valid",
                           kernel_regularizer=regularizer,
                           activation="tanh",
                           name="ga_conv_pool{}_ws{}".format(i, fsize))(embedding_layer)
                #X = BatchNormalization()(X)
                X = Pool(X)
            else:
                # NOTE: This will generate issues in the comparisson algorithms if
                # if fnum != embedding_dim
                X = Pool(embedding_layer)
            conv_out.append(Reshape((1, fnum))(X))

        if len(conv_out) == 1:
            outputs.append(conv_out[0])
        else:
            outputs.append(Concatenate(axis=1)(conv_out))

    return outputs


def group_b(embedding_layer, embedding_dim, max_seq_length, filters, poolings, conv2d_type, regularizer, cnn_type="Conv2D"):
    """ Returns a list with a tensor for each pooling function,
    where each tensor is of the shape:
    [?, embedding_dim, fnum]"""
    ###########
    # Group B #
    ###########
    # Expanding dim to use DepthwiseConv2D or Conv2D
    # new_embedding_layer = Reshape((max_seq_length, 1, embedding_dim))(embedding_layer)
    if conv2d_type:
        # Using Conv2D
        # Output shape: (?, max_seq_length, embedding_dim, 1)
        new_embedding_layer = Lambda(lambda x:K.expand_dims(x, axis=-1))(embedding_layer)
    else:
        # Using DepthwiseConv2D
        # Output shape: (?, max_seq_length, 1, embedding_dim)
        new_embedding_layer = Lambda(lambda x:K.expand_dims(x, axis=2))(embedding_layer)

    outputs = []
    for i, Pool in enumerate(poolings):
        conv_out = []
        for fsize, fnum in filters:
            if fsize:
                if conv2d_type:
                    # Output shape: (?, max_seq_length-fsize+1, embedding_dim, fnum)
                    X = Conv2D(fnum,
                               kernel_size=(fsize,1),
                               strides=(1,1),
                               padding="valid",
                               kernel_regularizer=regularizer,
                               activation="tanh",
                               name="gb_conv_pool{}_ws{}".format(i, fsize))(new_embedding_layer)
                else:
                    # Output shape: (?, max_seq_lenght, 1, embedding_dim * fnum)
                    X = DepthwiseConv2D(kernel_size=(fsize,1),
                                        strides=1,
                                        padding="valid",
                                        depth_multiplier=fnum,
                                        kernel_regularizer=regularizer,
                                        activation="tanh",
                                        name="gb_conv_pool{}_ws{}".format(i, fsize))(new_embedding_layer)
                    # Output shape: (?, max_seq_length-fszie+1, embedding_dim, fnum)
                    X = Reshape((-1, embedding_dim, fnum))(X)
                #X = BatchNormalization()(X)
                # Output shape: (?, embedding_dim, fnum)
                X = Pool(X)
            else:
                # To work, it needs fnum_ga = fnum_gb = embeddings_dim
                X = Pool(X)(embedding_layer)
            conv_out.append(Reshape((1, embedding_dim, fnum))(X))
        if len(conv_out) == 1:
            outputs.append(conv_out[0])
        else:
            outputs.append(Concatenate(axis=1)(conv_out))

    return outputs


def he_encoder(input_shape, filters_ga, filters_gb,
               conv2d_type, embeddings_dims, embeddings_matrices,
               word_to_index, max_seq_length,
               trainable_embeddings, use_groupa, use_groupb,
               regularizer, poolings_ga, poolings_gb):
    """input shape: (None, max_seq_length)
       embedding_layer shape: (None, max_seq_length, embedding_dim)
       None refers to the minibatch size."""
    # Input layer
    X_input = Input(input_shape)

    # Embedding layer which transforms sequence of indices to embeddings.
    # It can be set to update the embeddings (trainable parameter).
    # TODO: Use more than one embedding matrix
    embedding_layers = []
    for emb_dim, emb_matrix in zip(embeddings_dims, embeddings_matrices):
        emb_layer = Embedding(len(word_to_index)+1,
                              emb_dim,
                              weights=[emb_matrix],
                              input_length=max_seq_length,
                              trainable=trainable_embeddings)(X_input)
        embedding_layers.append(emb_layer)
    if len(embedding_layers) == 1:
        embedding_layer = embedding_layers[0]
    else:
        embedding_layer = Concatenate()(embedding_layers)
    #embedding_layer = BatchNormalization()(embedding_layer)
    embedding_dim = sum(embeddings_dims)

    assert use_groupa or use_groupb, "You must use Group A, B or both"

    if use_groupa and not use_groupb:
        ga_output = group_a(embedding_layer, filters_ga, poolings_ga, regularizer)
        my_model = Model(inputs=X_input,
                         outputs=ga_output,
                         name="he_model")
    elif not use_groupa and use_groupb:
        gb_output = group_b(embedding_layer, embedding_dim, max_seq_length, filters_gb, poolings_gb, conv2d_type, regularizer, "DepthwiseConv2D")
        my_model = Model(inputs=X_input,
                         outputs=gb_output,
                         name="he_model")
    elif use_groupa and use_groupb:
        ga_output = group_a(embedding_layer, filters_ga, poolings_ga, regularizer)
        gb_output = group_b(embedding_layer, embedding_dim, max_seq_length, filters_gb, poolings_gb, conv2d_type, regularizer, "DepthwiseConv2D")
        my_model = Model(inputs=X_input,
                         outputs=ga_output+gb_output,
                         name="he_model")
    return my_model


def algo1(s1_ga_pools, s2_ga_pools, use_cos, use_euc, use_abs):
    """:param s1_ga_pools: List of 'group A' outputs of sentece 1 for different
                     pooling types [max, min, avg] where each entry has shape
                     (?, len(filters_ga), fnum_ga)
       :param s2_ga_pools: List of 'group A' outputs of sentece 1 for different
                     pooling types [max, min, avg] where each entry has shape
                     (?, len(filters_ga), fnum_ga)
    """
    assert use_cos or use_euc, "You should use either cos or euc"
    res = []
    i = 0
    for s1_ga, s2_ga in zip(s1_ga_pools, s2_ga_pools):
        # Vector norms of len(filters_ga)-dimensional vectors
        # s1_norm.shape = s2_norm.shape = (?, len(filters_ga), fnum_ga)
        s1_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(s1_ga)
        s2_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(s2_ga)

        sims = []

        if use_cos:
            # Cosine Similarity between vectors of shape (len(filters_ga),)
            # cos_sim.shape = (?, fnum_ga)
            cos_sim = Lambda(lambda x: K.sum(x[0]*x[1], axis=1),
                             name="a1_{}pool_cos_ga".format(i))([s1_norm, s2_norm])
            sims.append(cos_sim)

        if use_euc:
            # Euclidean Distance between vectors of shape (len(filters_ga),)
            # euc_dis.shape = (?, fnum_ga)
            euc_dis = Lambda(lambda x: K.sqrt(K.clip(K.sum(K.square(x[0] - x[1]), axis=1), K.epsilon(), 1e+10)),
                             name="a1_{}pool_euc_ga".format(i))([s1_ga, s2_ga])
            sims.append(euc_dis)

        if use_abs:
            # Absolute Distance between vectors of shape (len(filters_ga),)
            # abs_dis.shape = (?, len(filters_ga), fnum_ga)
            abs_dis = Lambda(lambda x: K.abs(x[0] - x[1]),
                             name="a1_ga_{}pool_abs".format(i))([s1_ga, s2_ga])
            sims.append(Flatten()(abs_dis))

        if len(sims) == 1:
            res.append(sims[0])
        else:
            res.append(Concatenate()([cos_sim, euc_dis]))
        i += 1

    # feah = (3 * 2 * fnum_ga)
    if len(res) == 1:
        feah = res[0]
    else:
        feah = Concatenate(name="feah")(res)
    return feah


def algo2(s1_ga_pools, s1_gb_pools, s2_ga_pools, s2_gb_pools, use_cos, use_euc, use_abs):
    """:param s1_ga_pools: List of 'group A' outputs of sentece 1 for different
                     pooling types [max, min, avg] where each entry has shape
                     (?, len(filters_ga), fnum_ga)
       :param s2_ga_pools: List of 'group A' outputs of sentece 1 for different
                     pooling types [max, min, avg] where each entry has shape
                     (?, len(filters_ga), fnum_ga)
       :param s1_gb_pools: List of 'group B' outputs of sentence 1 for different
                     pooling types [max, min] where each entry has shape
                     (?, len(filters_gb), embeddings_dim, fnum_gb)
       :param s2_gb_pools: List of 'group B' outputs of sentence 2 for different
                     pooling types [max, min] where each entry has shape
                     (?, len(filters_gb), embeddings_dim, fnum_gb)
    """
    # First part of the algorithm using Group A outputs
    assert use_cos or use_euc or use_abs, "You should use either cos, euc or abs"
    res1 = []
    i = 0
    for s1_ga, s2_ga in zip(s1_ga_pools, s2_ga_pools):
        sims = []
        s1_ga_shape = s1_ga.get_shape().as_list()
        s2_ga_shape = s2_ga.get_shape().as_list()
        if use_cos:
            # Cosine similarity
            # Shape: cos_sim = (?, len(filters_ga), len(filters_ga))
            cos_sim = Dot(axes=2, normalize=True, name="a2_ga_{}pool_cos".format(i))([s1_ga, s2_ga])
            sims.append(Flatten()(cos_sim))

        if use_euc:
            # Euclidean distance
            # Shape: euc_dis = (?, len(filters_ga), len(filters_ga))
            s1_ga_bis = Reshape((s1_ga_shape[1], 1, s1_ga_shape[2]))(s1_ga)
            s2_ga_bis = Reshape((1, s2_ga_shape[1], s2_ga_shape[2]))(s2_ga)
            euc_dis = Lambda(lambda x: K.sqrt(K.clip(K.sum(K.square(x[0] - x[1]),
                            axis=-1, keepdims=False), K.epsilon(), 1e+10)),
                            name="a2_ga_{}pool_euc".format(i))([s1_ga_bis, s2_ga_bis])
            sims.append(Flatten()(euc_dis))

        if use_abs:
            # Shape: abs_dis = (?, len(filters_ga), len(filters_ga))
            s1_ga_bis = Reshape((s1_ga_shape[1], 1, s1_ga_shape[2]))(s1_ga)
            s2_ga_bis = Reshape((1, s2_ga_shape[1], s2_ga_shape[2]))(s2_ga)
            #abs_dis = Lambda(lambda x: K.sum(K.abs(K.clip(x[0] - x[1], 1e-7, 1e+10)), axis=-1, keepdims=False),
            # abs_dis = Lambda(lambda x: K.sum(K.abs(x[0] - x[1]), axis=-1, keepdims=False),
            #                  name="a2_ga_{}pool_abs".format(i))([s1_ga_bis, s2_ga_bis])
            abs_dis = Lambda(lambda x: K.abs(x[0] - x[1]),
                             name="a2_ga_{}pool_abs".format(i))([s1_ga_bis, s2_ga_bis])
            sims.append(Flatten()(abs_dis))
        if len(sims) == 1:
            res1.append(sims[0])
        else:
            res1.append(Concatenate()(sims))
        i += 1

    # Shape: feaa = (?, 3 * 3 * len(filters_ga) * len(filters_ga))
    if res1:
        if len(res1) == 1:
            feaa = res1[0]
        else:
            feaa = Concatenate(name="feaa")(res1)
    else:
        print("feaa is None")
        feaa = None

    # Second part of the algorithm using Group B outputs
    res2 = []
    i = 0
    for s1_gb, s2_gb in zip(s1_gb_pools, s2_gb_pools):
        sims = []
        if use_cos:
            # Vector norms of len(filters_gb)-dimensional vectors
            # s1_norm.shape = s2_norm.shape = (?, len(filters_gb), embedding_dim, fnum_gb)
            s1_norm = Lambda(lambda x:K.l2_normalize(x, axis=2), name="{}pool_s1_norm".format(i))(s1_gb)
            s2_norm = Lambda(lambda x:K.l2_normalize(x, axis=2), name="{}pool_s2_norm".format(i))(s2_gb)

            # Cosine Similarity between vectors of shape (embedding_dim,)
            # cos_sim.shape = (?, len(filters_gb) * fnum_gb)
            cos_sim = Flatten()(Lambda(lambda x: K.sum(x[0] * x[1], axis=2),
                                       name="a2_gb_{}pool_cos".format(i))([s1_norm, s2_norm]))
            sims.append(cos_sim)

        if use_euc:
            # Euclidean Distance between vectors of shape (embedding_dim,)
            # euc_dis.shape = (?, len(filters_gb) * fnum_gb)
            #euc_dis = Flatten()(Lambda(lambda x: K.sqrt(K.sum(K.square(K.clip(x[0] - x[1], 1e-7, 1e+10)),
            euc_dis = Flatten()(Lambda(lambda x: K.sqrt(K.clip(K.sum(K.square(x[0] - x[1]),
                                axis=2), K.epsilon(), 1e+10)), name="a2_gb_{}pool_euc".format(i))([s1_gb, s2_gb]))
            sims.append(euc_dis)

        if use_abs:
            # abs_dis.shape = (?, len(filters_gb) * embeddings_dim * fnum_gb)
            #abs_dis = Flatten()(Lambda(lambda x: K.sum(K.abs(K.clip(x[0] - x[1], 1e-7, 1e+10)),
            # abs_dis = Flatten()(Lambda(lambda x: K.sum(K.abs(x[0] - x[1]),
            #                     axis=2), name="a2_gb_{}pool_abs".format(i))([s1_gb, s2_gb]))
            abs_dis = Flatten()(Lambda(lambda x: K.abs(x[0] - x[1]),
                                name="a2_gb_{}pool_abs".format(i))([s1_gb, s2_gb]))
            sims.append(abs_dis)

        if len(sims) == 1:
            res2.append(sims[0])
        else:
            res2.append(Concatenate(axis=1)(sims))
        i += 1

    # feab = (?, 2 * (2 + embeddings_dim) * len(filters_gb) * fnum_gb)
    if res2:
        if len(res2) == 1:
            feab = res2[0]
        else:
            feab = Concatenate(name="feab")(res2)
    else:
        print("feab is None!")
        feab = None

    return feaa, feab


def he_model_siamese(input_shape, filters_ga, filters_gb,
             conv2d_type, embeddings_dims, embeddings_matrices,
             word_to_index, max_seq_length, reg_value,
             hidden_units, trainable_embeddings, use_groupa,
             use_groupb, use_algo1, use_algo2, poolings_ga, poolings_gb,
             use_cos_a1, use_cos_a2, use_euc_a1,
             use_euc_a2, use_abs_a1, use_abs_a2):
    # TODO: Add docstring and comments

    assert use_algo1 or use_algo2, "You must use Algorithm 1, 2 or both"

    # Allowed combinations of grouptype and algtype
    assert not(use_groupa and use_groupb and use_algo1 and not use_algo2), \
            "Not a valid combination of groups and algorithms. Group B computed but not used. Algorithm 1 only uses Group A."
    assert not(not use_groupa and use_groupb and use_algo1 and use_algo2), \
            "Not a valid combination of groups and algorithms. Algoritm 1 needs Group A."
    assert not(not use_groupa and use_groupb and use_algo1 and not use_algo2), \
            "Not a valid combination of groups and algorithms. Group B needs Algorithm 2 while Algoritm 1 needs Group A."

    # Defining regularizer
    if reg_value != None:
        regularizer = regularizers.l2(reg_value)
    else:
        regularizer = None

    # Generating encoder
    base_model = he_encoder((max_seq_length, ), filters_ga, filters_gb,
                            conv2d_type, embeddings_dims, embeddings_matrices,
                            word_to_index, max_seq_length,
                            trainable_embeddings, use_groupa, use_groupb,
                            regularizer, poolings_ga, poolings_gb)

    # Defining inputs
    X1_input = Input(input_shape, name="input_X1")
    X2_input = Input(input_shape, name="input_X2")

    # Encoding inputs
    encoded_1 = base_model(X1_input)
    encoded_2 = base_model(X2_input)
    if not isinstance(encoded_1, list):
        print("Not a list instance. Transforming to list!")
        encoded_1 = [encoded_1]
        encoded_2 = [encoded_2]

    if use_groupa and not use_groupb:
        s1_ga_pools = encoded_1
        s2_ga_pools = encoded_2
        s1_gb_pools = []
        s2_gb_pools = []
    elif not use_groupa and use_groupb:
        s1_ga_pools = []
        s2_ga_pools = []
        s1_gb_pools = encoded_1
        s2_gb_pools = encoded_2
    elif use_groupa and use_groupb:
        s1_ga_pools = encoded_1[:len(poolings_ga)]
        s2_ga_pools = encoded_2[:len(poolings_ga)]
        s1_gb_pools = encoded_1[len(poolings_ga):]
        s2_gb_pools = encoded_2[len(poolings_ga):]

    if use_algo1 and not use_algo2:
        feah = algo1(s1_ga_pools, s2_ga_pools, use_cos_a1, use_euc_a1, use_abs_a1)
        feats = feah
    elif not use_algo1 and use_algo2:
        feaa, feab = algo2(s1_ga_pools, s1_gb_pools, s2_ga_pools, s2_gb_pools, use_cos_a2, use_euc_a2, use_abs_a2)
        if use_groupa and not use_groupb:
            # Means feab = None
            feats = feaa
        elif not use_groupa and use_groupb:
            # Means feaa = None
            feats = feab
        else:
            feats = Concatenate(name="feats")([feaa, feab])
    elif use_algo1 and use_algo2:
        feah = algo1(s1_ga_pools, s2_ga_pools, use_cos_a1, use_euc_a1, use_abs_a1)
        feaa, feab = algo2(s1_ga_pools, s1_gb_pools, s2_ga_pools, s2_gb_pools, use_cos_a2, use_euc_a2, use_abs_a2)
        if use_groupa and not use_groupb:
            # Means feab = None
            feats = Concatenate(name="feats")([feah, feaa])
        elif use_groupa and use_groupb:
            feats = Concatenate(name="feats")([feah, feaa, feab])

    X = Dense(hidden_units, name="fully_connected",
              kernel_regularizer=regularizer, activation="tanh")(feats)
    X = Dense(2, name="output", kernel_regularizer=regularizer, activation="softmax")(X)

    siamese_net = Model(inputs=[X1_input, X2_input], outputs=X, name="he_model_siamese")
    return siamese_net, base_model

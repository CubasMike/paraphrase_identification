import os

# Needed for reproducibility
os.environ["PYTHONHASHSEED"] = "0"

# No GPU. Needed for reproducibility
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Needed environment variable for using GPU
# ItIDs match nvidia-smi
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 

# Manually selecting a GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple

# Needed for reproducibility
import numpy as np
import random as rn
# np.random.seed(1)
# rn.seed(2)

# Needed for reproducibility
# Specific backend
# import tensorflow as tf
# No multithreading
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# Needed for reproducibility
# Specific backend
# tf.set_random_seed(1234)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping, TerminateOnNaN, ModelCheckpoint
from keras.layers import Lambda, GlobalMaxPooling1D
from keras.models import load_model
from future_keras import DepthwiseConv2D

from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import pickle
import mpcnn_model
import utils
import argparse
import datetime

def argument_parser():
    parser = argparse.ArgumentParser(description="""Multi-Perspective Sentence Similarity applied to Paraphrase Identification.
            The model is described in: He, H., Gimpel, K., Lin, J.J.: Multi-perspective sentence similarity modeling with convolutional neural networks. In: Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. pp. 1576–1586. Association for Computational Linguistics, Lisbon, Portugal (September 2015)

            When using this code please cite: Sanchez-Perez, M.A. 2018 "Plagiarism Detection through Paraphrase Identification", PhD thesis, Centro de Investigación en Computación, Instituto Politécnico Nacional, Mexico City, Mexico.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version",
            type=str,
            dest="version",
            help="Version used to pre-process the data and construct the embedding matrices. Check the date on the data available.")

    parser.add_argument("--preproc",
            type=str,
            dest="pp_name",
            help="Tokenizer used. Check the name of the files available in <data>.")

    parser.add_argument("--cap",
            type=str,
            dest="lower_opt",
            help="Capitalziation option when preprocessing. Check the name of the files available in <data>.")

    parser.add_argument("--embcap",
            type=str,
            dest="emb_opt",
            help="Lookup technique used when retrieving the word embeddings.")

    parser.add_argument("--rev-train",
            dest="reversed_train",
            action="store_true",
            help="""Duplicate the training data size by adding the reverse version of the pair of sentences.
                    Helpful when not using a Siamese model.""")
    parser.add_argument("--no-rev-train",
                        dest="reversed_train",
                        action="store_false")

    parser.add_argument("--ngrams",
            type=int,
            dest="ngrams",
            help="""Defines the filters sizes from 1 to the assigned value.
                    Eg. 1,2,3 in the case of ngrams=3.""")

    parser.add_argument("--fnum-ga",
            type=int,
            dest="fnum_ga",
            help="Number of holistic filters (Group A).")

    parser.add_argument("--inf-ga",
            dest="use_inf_ga",
            action="store_true",
            help="""Defines if using the inf filter size in the holistic filters (Group A).
                    When used, the number of filters must be the same as the embeddings size
                    because computing similarity requires vectors of the same dimensions.""")
    parser.add_argument("--no-inf-ga",
            dest="use_inf_ga",
            action="store_false")

    parser.add_argument("--fnum-gb",
            type=int,
            dest="fnum_gb",
            help="Number of per-dimension filters (Group B).")

    parser.add_argument("--inf-gb",
            dest="use_inf_gb",
            action="store_true",
            help="""Defines if using the inf filter size in the per-dimension filters (Group B).
                    When used, the number of filters must be the same as the embeddings size
                    because computing similarity requires vectors of the same dimensions.""")
    parser.add_argument("--no-inf-gb",
            dest="use_inf_gb",
            action="store_false")

    parser.add_argument("--conv2d",
            dest="conv2d_type",
            action="store_true",
            help="""Defines if using tradition spatial Conv2D or independent Conv2D
                    (DepthwiseConv2D) for each dimension of the embeddings
                    (as stated in their paper).""")
    parser.add_argument("--depthwise-conv2d",
            dest="conv2d_type",
            action="store_false")

    parser.add_argument("--pool-ga",
            nargs="+",
            choices=("max", "min", "mean"),
            dest="poolings_ga",
            help="Pooling operators used in group A.")

    parser.add_argument("--pool-gb",
            nargs="+",
            choices=("max", "min", "mean"),
            dest="poolings_gb",
            help="Pooling operators used in group B.")

    parser.add_argument("--epochs",
            type=int,
            dest="epochs",
            help="Number of training epochs.")

    parser.add_argument("--bsize",
            type=int,
            dest="batch_size",
            help="Batch size used in gradient descent.")

    parser.add_argument("--train-emb",
            dest="trainable_embeddings",
            action="store_true",
            help="Training the word embeddings as part of the model.")
    parser.add_argument("--no-train-emb",
            dest="trainable_embeddings",
            action="store_false")

    parser.add_argument("--hid-units",
            type=int,
            dest="hidden_units",
            help="Hidden units used in the similarity layer.")

    parser.add_argument("--reg",
            type=float,
            dest="reg_value",
            help="L2 regulatization value.")

    parser.add_argument("--groupa",
            dest="use_groupa",
            action="store_true",
            help="""Defines if using group A. Check authors original paper for detailed description
                  or check the code of the model.""")
    parser.add_argument("--no-groupa",
            dest="use_groupa",
            action="store_false")

    parser.add_argument("--groupb",
            dest="use_groupb",
            action="store_true",
            help="""Defines if using group B. Check authors original paper for detailed description
                  or check the code of the model.""")
    parser.add_argument("--no-groupb",
            dest="use_groupb",
            action="store_false")

    parser.add_argument("--algo1",
            dest="use_algo1",
            action="store_true",
            help="""Define if using algorithm 1.
                    Check authors original paper for detailed description
                    or check the code of the model.""")
    parser.add_argument("--no-algo1",
            dest="use_algo1",
            action="store_false")

    parser.add_argument("--algo2",
            dest="use_algo2",
            action="store_true",
            help="""Define if using algorithm 2.
                    Check authors original paper for detailed description
                    or check the code of the model.""")
    parser.add_argument("--no-algo2",
            dest="use_algo2",
            action="store_false")

    parser.add_argument("--cos-a1",
            dest="use_cos_a1",
            action="store_true",
            help="Defines if using cosine similarity in algorithm 1.")
    parser.add_argument("--no-cos-a1",
            dest="use_cos_a1",
            action="store_false")

    parser.add_argument("--euc-a1",
            dest="use_euc_a1",
            action="store_true",
            help="Defines if using Euclidean distance in algorithm 1.")
    parser.add_argument("--no-euc-a1",
            dest="use_euc_a1",
            action="store_false")

    parser.add_argument("--abs-a1",
            dest="use_abs_a1",
            action="store_true",
            help="Defines if using absolute difference in algorithm 1.")
    parser.add_argument("--no-abs-a1",
            dest="use_abs_a1",
            action="store_false")

    parser.add_argument("--cos-a2",
            dest="use_cos_a2",
            action="store_true",
            help="Defines if using cosine similarity in algorithm 2.")
    parser.add_argument("--no-cos-a2",
            dest="use_cos_a2",
            action="store_false")

    parser.add_argument("--euc-a2",
            dest="use_euc_a2",
            action="store_true",
            help="Defines if using Euclidean distance in algorithm 2.")
    parser.add_argument("--no-euc-a2",
            dest="use_euc_a2",
            action="store_false")

    parser.add_argument("--abs-a2",
            dest="use_abs_a2",
            action="store_true",
            help="Defines if using absolute difference in algorithm 2.")
    parser.add_argument("--no-abs-a2",
            dest="use_abs_a2",
            action="store_false")

    parser.add_argument("--autoneg",
            type=int,
            dest="autoneg",
            help="Number of automatically generated negative samples.")

    parser.add_argument("--embeddings",
            nargs="+",
            choices=("glove", "POSword2vec", "paragram25"),
            dest="embeddings",
            help="""Word embeddings being used. Check the name of the files available in <data>.
                    Embeddings will be concatenated.""")

    parser.add_argument("--optimizer",
            type=str,
            choices=("adam", "sgd"),
            dest="optimizer",
            help="Optimizer used during back propagation.")

    parser.add_argument("--learning-rate",
            type=float,
            dest="lr",
            help="Learning rate used by the optimizer.")

    parser.add_argument("--loss",
            type=str,
            choices=("categorical_crossentropy", "categorical_hinge", "kullback_leibler_divergence"),
            dest="loss",
            help="Loss used during backpropagation.")

    parser.add_argument("--print-encoder",
            dest="print_encoder",
            action="store_true",
            help="Defines if printing encoder model.")
    parser.add_argument("--no-print_encoder",
            dest="print_encoder",
            action="store_false")

    parser.add_argument("--print-model",
            dest="print_model",
            action="store_true",
            help="Defines if printing general model.")
    parser.add_argument("--no-print_model",
            dest="print_model",
            action="store_false")

    # Default parameters
    parser.set_defaults(# Related to preprocessed MRSPC data
                        version="20180611",
                        pp_name="POSkimprep",
                        lower_opt="nolower",
                        emb_opt="embcap",
                        # Embeddings already extracted for MSRPC
                        embeddings=["glove", "POSword2vec", "paragram25"],

                        # Related to dataset generation
                        autoneg=0,
                        reversed_train=False,

                        # Related to the model
                        ngrams=3,
                        epochs=10,
                        batch_size=32,
                        trainable_embeddings=True,
                        hidden_units=250,
                        reg_value=1e-4,
                        optimizer="adam",
                        lr=0.001,
                        loss="categorical_crossentropy",
                        # Group A
                        fnum_ga=525,
                        use_inf_ga=False,
                        use_groupa=True,
                        poolings_ga=["max"],
                        # Group B 
                        fnum_gb=20,
                        use_inf_gb=False,
                        poolings_gb=["max"],
                        use_groupb=True,
                        conv2d_type=True,
                        # Algorithm 1
                        use_algo1=True,
                        use_cos_a1=True,
                        use_euc_a1=True,
                        use_abs_a1=True,
                        # Algorithm 2
                        use_algo2=True,
                        use_cos_a2=True,
                        use_euc_a2=True,
                        use_abs_a2=True,

                        # Others
                        print_encoder=False,
                        print_model=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    print("\n")

    # (None, args.holistic_fnum) represents the inf filter size. There is going to be an error
    # if holistic_fnum != embeddings_size due to dimensions missmatching during concatenation
    filters_ga = [(i+1, args.fnum_ga) for i in range(args.ngrams)]
    if args.use_inf_ga:
        filters_ga.append((None, args.fnum_ga))
    filters_gb = [(i+1, args.fnum_gb) for i in range(args.ngrams)]
    if args.use_inf_gb:
        filters_gb.append((None, args.fnum_gb))

    # Generating the pooling layers
    poolings_ga = []
    if "max" in args.poolings_ga:
        poolings_ga.append(Lambda(lambda x: K.max(x, axis=1), name="ga_maxpool"))
    if "min" in args.poolings_ga:
        poolings_ga.append(Lambda(lambda x: K.min(x, axis=1), name="ga_minpool"))
    if "mean" in args.poolings_ga:
        #poolings_ga.append(Lambda(lambda x: K.sum(x, axis=1)/K.sum(K.cast(K.not_equal(x, 0), "float32"), axis=1) , name="ga_meanpool"))
        poolings_ga.append(Lambda(lambda x: K.mean(x, axis=1), name="ga_meanpool"))

    poolings_gb = []
    if "max" in args.poolings_gb:
        poolings_gb.append(Lambda(lambda x: K.max(x, axis=1), name="gb_maxpool"))
    if "min" in args.poolings_gb:
        poolings_gb.append(Lambda(lambda x: K.min(x, axis=1), name="gb_minpool"))
    if "mean" in args.poolings_gb:
        #poolings_gb.append(Lambda(lambda x: K.sum(x, axis=1)/K.sum(K.cast(K.not_equal(x, 0), "float32"), axis=1) , name="gb_meanpool"))
        poolings_gb.append(Lambda(lambda x: K.mean(x, axis=1), name="gb_meanpool"))

    # Loading embeddings matrices
    print("=======================\nLoading embedding files\n=======================")
    embedding_dims = []
    embedding_matrices = []
    if "glove" in args.embeddings:
        emb_fn = "msrpc_{}_{}_{}_{}_{}.pickle".format(args.pp_name, args.lower_opt, "glove", args.emb_opt, args.version)
        print("loading ...", emb_fn)
        [embeddings_matrix, unknown_words] = pickle.load(open("./data/"+emb_fn, 'rb'))
        print("Embeddings shape (pretrained GloVe): {}  unknown tokens: {}".format(embeddings_matrix.shape, len(unknown_words)))
        embedding_dims.append(embeddings_matrix.shape[1])
        embedding_matrices.append(embeddings_matrix)

    if "POSword2vec" in args.embeddings:
        emb_fn = "msrpc_{}_{}_{}_{}_{}.pickle".format(args.pp_name, args.lower_opt, "POSword2vec", args.emb_opt, args.version)
        print("loading ...", emb_fn)
        [embeddings_matrix, unknown_words] = pickle.load(open("./data/"+emb_fn, 'rb'))
        print("Embeddings shape (pretrained POS word2vec): {}  unknown tokens: {}".format(embeddings_matrix.shape, len(unknown_words)))
        embedding_dims.append(embeddings_matrix.shape[1])
        embedding_matrices.append(embeddings_matrix)

    if "paragram25" in args.embeddings:
        emb_fn = "msrpc_{}_{}_{}_{}_{}.pickle".format(args.pp_name, args.lower_opt, "paragram25", args.emb_opt, args.version)
        print("loading ...", emb_fn)
        [embeddings_matrix, unknown_words] = pickle.load(open("./data/"+emb_fn, 'rb'))
        print("Embeddings shape (pretrained PARAGRAM): {}  unknown tokens: {}".format(embeddings_matrix.shape, len(unknown_words)))
        embedding_dims.append(embeddings_matrix.shape[1])
        embedding_matrices.append(embeddings_matrix)

    print("Final embeddings dim:", sum(embedding_dims))
    print("\n")

    # Generating datasets from parsed MSRPC
    print("===================\nGenerating datasets\n===================")
    (index_to_word, word_to_index,
     X_train1, X_train2, Y_train,
     X_test1, X_test2, Y_test) = utils.generate_dataset(args.pp_name, args.lower_opt, args.version,
                                          max_seq_length=-1, reverse_train_pairs=args.reversed_train,
                                          padding=True, autoneg=args.autoneg)

    max_seq_length = X_train1.shape[1]
    print("Max seq length:", max_seq_length)
    print("X_train:", X_train1.shape)
    print("Y_train:", Y_train.shape)
    print("X_test:", X_test1.shape)
    print("Y_test:", Y_test.shape)
    print("\n")

    # Generating the model
    print("================\nGenerating model\n================",)
    print("Group A: {}   Group B: {}".format(args.use_groupa, args.use_groupb))
    if args.use_groupa:
        print("Group A config\n--------------")
        print("Filters:", filters_ga)
        print("Pool ops:", args.poolings_ga)
        print()
    if args.use_groupb:
        print("Group B config\n--------------")
        print("Using {}".format("Conv2D" if args.conv2d_type else "DepthwiseConv2D"))
        print("Filters:", filters_gb)
        print("Pool ops:", args.poolings_gb)
        print()

    print("Algorithm 1: {}   Algorithm 2: {}".format(args.use_algo1, args.use_algo2))
    if args.use_algo1:
        print("Algorithm 1\n-----------")
        print("Cosine similarity:", args.use_cos_a1)
        print("Euclidean distance:", args.use_euc_a1)
        print("Absolute difference:", args.use_abs_a1)
        print()
    if args.use_algo2:
        print("Algorithm 2\n-----------")
        print("Cosine similarity:", args.use_cos_a2)
        print("Euclidean distance:", args.use_euc_a2)
        print("Absolute difference:", args.use_abs_a2)
        print()

    print("Other settings\n--------------")
    print("MLP hidden units:", args.hidden_units)
    print("Regularization:", args.reg_value)
    print("Trainable embeddings:", args.trainable_embeddings)
    print()

    model = mpcnn_model.he_model_siamese
    siamese_model, encoder = model((max_seq_length, ), filters_ga, filters_gb,
                          args.conv2d_type, embedding_dims, embedding_matrices,
                          word_to_index, max_seq_length, args.reg_value,
                          args.hidden_units, args.trainable_embeddings,
                          args.use_groupa, args.use_groupb, args.use_algo1,
                          args.use_algo2, poolings_ga, poolings_gb,
                          args.use_cos_a1, args.use_cos_a2, args.use_euc_a1,
                          args.use_euc_a2, args.use_abs_a1, args.use_abs_a2)
    print("Done!")
    print("\n")

    # Printing summaries
    if args.print_encoder:
        encoder.summary()
    if args.print_model:
        siamese_model.summary()

    # Compiling model
    print("===============\nCompiling model\n===============")
    print("Optimizer:", args.optimizer)
    print("Learning rate:", args.lr)
    print("Loss:", args.loss)
    if args.optimizer == "adam":
        optimizer = optimizers.Adam(lr=args.lr)#, clipnorm=1.)
    if args.optimizer == "sgd":
        optimizer = optimizers.SGD(lr=args.lr)
    siamese_model.compile(optimizer=optimizer,
            loss=args.loss,
            metrics=["accuracy"])
    print("Done!")
    print("\n")

    # Training model
    print("==================\nTraining model ...\n==================")
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    # Early stopping
    early_stopping_patience = 5
    early_stop = EarlyStopping(monitor='loss',
                               min_delta=0,
                               patience=early_stopping_patience,
                               verbose=0,
                               mode='auto')

    # Terminate training when NaN loss is encountered
    stop_nan = TerminateOnNaN()

    # Callbacks
    early_stop = False
    my_calls = [stop_nan]
    if early_stop:
        my_calls.append(early_stop)

    use_class_weight = False if args.autoneg == 0 else True
    if use_class_weight:
        if Y_train[Y_train == 1].size > Y_train[Y_train == 0].size:
            class_weight = {1:1.0, 0: Y_train[Y_train == 1].size / Y_train[Y_train == 0].size}
        else:
            class_weight = {1:Y_train[Y_train == 0].size / Y_train[Y_train == 1].size, 0: 1.0}
        print("class_weight", class_weight)
    else:
        class_weight = None

    # Initial prediction with random initialization
    print("Forward propagation with random values")
    pred=siamese_model.predict([X_test1, X_test2])
    predclass=np.argmax(pred, axis=1)
    acc = accuracy_score(Y_test, predclass)
    print(acc)
    f1 = f1_score(Y_test, predclass)
    print(f1)
    cm = confusion_matrix(Y_test, predclass)
    print(cm)

    # Date used to keep track of experiments
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpointer = ModelCheckpoint(filepath="../checkpoints/he_mpcnn/cp_{}.hdf5".format(date), verbose=1, save_best_only=True, monitor="val_acc", mode="max")
    my_calls.append(checkpointer)

    history = siamese_model.fit(x=[X_train1, X_train2],
                                y=to_categorical(Y_train),
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                #validation_data=([X_test1, X_test2], to_categorical(Y_test)),
                                validation_split=0.2,
                                class_weight=class_weight,
                                callbacks=my_calls,
                                verbose=2)
    print("Done!")
    print("\n")
    # Temporarily printing some weights
    # W1 = encoder.get_layer("ga_conv_pool0_ws3").get_weights()
    # print("ga_conv_0p_ws3 PARAMETERS\n", W1)
    # print()
    # 
    # tmp_layer = siamese_model.get_layer("output")
    # W2 = tmp_layer.get_weights()
    # print("output layer weights\n", W2)
    # print()

    # Evaluating
    #print(siamese_model.evaluate(x=[X_test1, X_test2], y=to_categorical(Y_test)))
    print("======================\nLast epoch predictions\n======================")
    pred = siamese_model.predict(x=[X_test1, X_test2])
    predclass=np.argmax(pred, axis=1)
    test_acc = accuracy_score(Y_test, predclass)
    test_f1 = f1_score(Y_test, predclass)
    test_confusion_matrix = confusion_matrix(Y_test, predclass)
    print("acc:", test_acc)
    print("f1: ", test_f1)
    print("confusion matrix")
    print(test_confusion_matrix)
    history.history["last_epoch_pred"] = pred
    history.history["last_epoch_test_acc"] = test_acc
    history.history["last_epoch_test_f1"] = test_f1
    history.history["last_epoch_test_confusion_matrix"] = test_confusion_matrix

    print("======================\nBest epoch predictions\n======================")
    # Loading best epoch model
    if args.conv2d_type:
        best_model = load_model("../checkpoints/he_mpcnn/cp_{}.hdf5".format(date))
    else:
        best_model = load_model("../checkpoints/he_mpcnn/cp_{}.hdf5".format(date),
                custom_objects={"DepthwiseConv2D":DepthwiseConv2D})
    pred = best_model.predict(x=[X_test1, X_test2])
    predclass=np.argmax(pred, axis=1)
    test_acc = accuracy_score(Y_test, predclass)
    test_f1 = f1_score(Y_test, predclass)
    test_confusion_matrix = confusion_matrix(Y_test, predclass)
    print("acc:", test_acc)
    print("f1: ", test_f1)
    print("confusion matrix")
    print(test_confusion_matrix)
    history.history["best_epoch_pred"] = pred
    history.history["best_epoch_test_acc"] = test_acc
    history.history["best_epoch_test_f1"] = test_f1
    history.history["best_epoch_test_confusion_matrix"] = test_confusion_matrix

    with open("../runs/he_mpcnn/run_{}_he_approach.p".format(date), "wb") as fid:
        pickle.dump([history.history, args], fid)

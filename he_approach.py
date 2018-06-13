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
np.random.seed(1)
rn.seed(2)

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
from keras.callbacks import Callback, EarlyStopping, TerminateOnNaN
from keras.layers import Lambda, GlobalMaxPooling1D

from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import pickle
import mpcnn_model
import utils
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description="Multi-Perspective Sentence Similarity applied to Paraphrase Identification.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--version",
            type=str,
            default="20180611",
            dest="version",
            help="Version used to pre-process the data and construct the embedding matrices. Check the date on the data available.")

    parser.add_argument("--preproc",
            type=str,
            default="POSkimprep",
            dest="pp_name",
            help="Tokenizer used. Check the name of the files available in <data>.")

    parser.add_argument("--capitalization",
            type=str,
            default="nolower",
            dest="lower_opt",
            help="Capitalziation option when preprocessing. Check the name of the files available in <data>.")

    parser.add_argument("--embcap",
            type=str,
            default="embcap",
            dest="emb_opt",
            help="Lookup technique used when retrieving the word embeddings.")

    parser.add_argument("--no-reverse-train",
                        dest="reversed_train",
                        action="store_false")
    parser.add_argument("--reverse_train",
            dest="reversed_train",
            action="store_true",
            help="""Duplicate the training data size by adding the reverse version of the pair of sentences.
                    Helpful when not using a Siamese model.""")
    parser.set_defaults(reversed_train=False)


    parser.add_argument("--ngrams",
            type=int,
            default=3,
            dest="ngrams",
            help="""Defines the filters sizes from 1 to the assigned value.
                    Eg. 1,2,3 in the case of ngrams=3.""")

    parser.add_argument("--holistic_filters",
            type=int,
            default=525,
            dest="holistic_fnum",
            help="Number of holistic filters.")

    parser.add_argument("--inf",
            dest="use_inf",
            action="store_true")
    parser.add_argument("--no-inf",
            dest="use_inf",
            action="store_false",
            help="Defines if using the inf filter size in the holistic filters")
    parser.set_defaults(use_inf=True)

    parser.add_argument("--perdimension_filters",
            type=int,
            default=20,
            dest="perdim_fnum",
            help="Number of per-dimension filters.")

    parser.add_argument("--poolings_group_a",
            nargs="+",
            default=["max", "min", "avg"],
            choices=("max", "min", "avg"),
            dest="poolings_ga",
            help="Pooling operators used in group A.")

    parser.add_argument("--poolings_group_b",
            nargs="+",
            default=["max", "min"],
            choices=("max", "min", "avg"),
            dest="poolings_gb",
            help="Pooling operators used in group B.")

    parser.add_argument("--epochs",
            type=int,
            default=30,
            dest="epochs",
            help="Number of training epochs.")

    parser.add_argument("--batch_size",
            type=int,
            default=32,
            dest="batch_size",
            help="Batch size used in gradient descent.")

    parser.add_argument("--trainable_embeddings",
            dest="trainable_embeddings",
            action="store_true",
            help="Training the word embeddings as part of the model.")
    parser.add_argument("--no-trainable_embeddings",
            dest="trainable_embeddings",
            action="store_false")
    parser.set_defaults(trainable_embeddings=True)

    parser.add_argument("--hidden_units",
            type=int,
            default=250,
            dest="hidden_units_merger",
            help="Hidden units used in the similarity layer.")

    parser.add_argument("--regularization",
            type=float,
            default=1e-4,
            dest="regularization_param",
            help="L2 regulatization value.")

    parser.add_argument("--groups",
            type=str,
            default="both",
            dest="grouptype",
            choices=("group_a_only", "group_b_only", "both"),
            help="""Which group to use. Check authors original paper for detailed description
                  or check the code of the model.""")

    parser.add_argument("--algorithms",
            type=str,
            default="both",
            dest="algtype",
            choices=("algo1_only", "algo2_only", "both"),
            help="""Which algorithm to use to compared certain regions of the sentence representations.
                    Check authors original paper for detailed description
                    or check the code of the model.""")

    parser.add_argument("--cos",
            dest="use_cos",
            action="store_true",
            help="Defines if using cosine similarity in the algoritms.")
    parser.add_argument("--no-cos",
            dest="use_cos",
            action="store_false")
    parser.set_defaults(use_cos=True)

    parser.add_argument("--euc",
            dest="use_euc",
            action="store_true",
            help="Defines if using Euclidean distance in the algorithms.")
    parser.add_argument("--no-euc",
            dest="use_euc",
            action="store_false")
    parser.set_defaults(use_euc=True)

    parser.add_argument("--abs",
            dest="use_abs",
            action="store_true",
            help="Defines if using absolute difference in the algortims.")
    parser.add_argument("--no-abs",
            dest="use_abs",
            action="store_false")
    parser.set_defaults(use_abs=True)

    parser.add_argument("--autoneg",
            type=int,
            default=0,
            dest="autoneg",
            help="Number of automatically generated negative samples.")

    parser.add_argument("--embeddings",
            nargs="+",
            default=["glove", "POSword2vec", "paragram25"],
            choices=("glove", "POSword2vec", "paragram25"),
            dest="embeddings",
            help="""Word embeddings being used. Check the name of the files available in <data>.
                    Embeddings will be concatenated.""")

    parser.add_argument("--optimizer",
            type=str,
            default="adam",
            choices=("adam", "sgd"),
            dest="optimizer",
            help="Optimizer used during back propagation.")

    parser.add_argument("--learning-rate",
            type=float,
            default=0.001,
            dest="lr",
            help="Learning rate used by the optimizer.")

    parser.add_argument("--loss",
            type=str,
            default="categorical_crossentropy",
            choices=("categorical_crossentropy", "categorical_hinge", "kullback_leibler_divergence"),
            dest="loss",
            help="Loss used during backpropagation.")

    parser.add_argument("--print_encoder",
            dest="print_encoder",
            action="store_true",
            help="Defines if printing encoder model.")
    parser.add_argument("--no-print_encoder",
            dest="print_encoder",
            action="store_false")
    parser.set_defaults(print_encoder=False)

    parser.add_argument("--print_model",
            dest="print_model",
            action="store_true",
            help="Defines if printing general model.")
    parser.add_argument("--no-print_model",
            dest="print_model",
            action="store_false")
    parser.set_defaults(print_model=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    print("\n")

    # (None, args.holistic_fnum) represents the inf filter size. There is going to be an error
    # if holistic_fnum != embeddings_size due to dimensions missmatching during concatenation
    filters_ga = [(i+1, args.holistic_fnum) for i in range(args.ngrams)]
    if args.use_inf:
        filters_ga.append((None, args.holistic_fnum))
    filters_gb = [(i+1, args.perdim_fnum) for i in range(args.ngrams)]
    model = mpcnn_model.he_model_siamese

    # Generating the pooling layers
    poolings_ga = []
    if "max" in args.poolings_ga:
        poolings_ga.append(Lambda(lambda x: K.max(x, axis=1), name="ga_maxpool"))
    if "min" in args.poolings_ga:
        poolings_ga.append(Lambda(lambda x: K.min(x, axis=1), name="ga_minpool"))
    if "avg" in args.poolings_ga:
        #poolings_ga.append(Lambda(lambda x: K.sum(x, axis=1)/K.sum(K.cast(K.not_equal(x, 0), "float32"), axis=1) , name="ga_avgpool"))
        poolings_ga.append(Lambda(lambda x: K.mean(x, axis=1), name="ga_avgpool"))

    poolings_gb = []
    if "max" in args.poolings_gb:
        poolings_gb.append(Lambda(lambda x: K.max(x, axis=1), name="gb_maxpool"))
    if "min" in args.poolings_gb:
        poolings_gb.append(Lambda(lambda x: K.min(x, axis=1), name="gb_minpool"))
    if "avg" in args.poolings_gb:
        #poolings_gb.append(Lambda(lambda x: K.sum(x, axis=1)/K.sum(K.cast(K.not_equal(x, 0), "float32"), axis=1) , name="gb_avgpool"))
        poolings_gb.append(Lambda(lambda x: K.mean(x, axis=1), name="gb_avgpool"))

    # Loading embeddings matrices
    print("Loading embedding files\n-----------------------")
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
    print("Generating datasets\n-------------------")
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
    print("Generating model...\n-------------------",)
    print("Using groups:", args.grouptype)
    if args.grouptype == "both" or args.grouptype == "group_a_only":
        print("Group A config\n--------------")
        print("Filters:", filters_ga)
        print("Pool ops:", args.poolings_ga)
        print()
    if args.grouptype == "both" or args.grouptype == "group_b_only":
        print("Group B config\n--------------")
        print("Filters:", filters_gb)
        print("Pool ops:", args.poolings_gb)
        print()

    print("Using algorithms:", args.algtype)
    print("Using metrics: {} {} {}".format("cos" if args.use_cos else "",
                                           "euc" if args.use_euc else "",
                                           "abs" if args.use_abs else ""))

    print("Other settings\n--------------")
    print("MLP hidden units:", args.hidden_units_merger)
    print("Regularization:", args.regularization_param)
    print("Trainable embeddings:", args.trainable_embeddings)

    siamese_model, encoder = model((max_seq_length, ), filters_ga, filters_gb,
                          embedding_dims, embedding_matrices,
                          word_to_index, max_seq_length, args.regularization_param,
                          args.hidden_units_merger, args.trainable_embeddings,
                          args.grouptype, args.algtype, poolings_ga, poolings_gb,
                          args.use_cos, args.use_euc, args.use_abs)
    print("Done!")
    print("\n")

    # Printing summaries
    if args.print_encoder:
        encoder.summary()
    if args.print_model:
        siamese_model.summary()

    # Compiling model
    print("Compiling model ...\n-------------------")
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

    # Callback to store prediction scores for each epoch
    class prediction_history(Callback):
        def __init__(self):
            self.acchis = []
            self.f1his = []
            self.cmhis = []
        def on_epoch_end(self, epoch, logs={}):
            pred=self.model.predict([X_test1, X_test2])
            predclass=np.argmax(pred, axis=1)
            acc = accuracy_score(Y_test, predclass)
            print(acc)
            self.acchis.append(acc)
            f1 = f1_score(Y_test, predclass)
            print(f1)
            self.f1his.append(f1)
            cm = confusion_matrix(Y_test, predclass)
            print(cm)
            self.cmhis.append(cm)

    per_epoch_preds = prediction_history()

    # Training model
    print("Training model ...\n------------------")
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    # Early stopping
    early_stopping_patience = 10
    early_stop = EarlyStopping(monitor='loss',
                               min_delta=0,
                               patience=early_stopping_patience,
                               verbose=0,
                               mode='auto')

    # Terminate training when NaN loss is encountered
    stop_nan = TerminateOnNaN()

    # Used callbacks
    my_calls = [per_epoch_preds, stop_nan]#None#[early_stop]

    use_class_weight = False if args.autoneg == 0 else True
    if use_class_weight:
        if Y_train[Y_train == 1].size > Y_train[Y_train == 0].size:
            class_weight = {1:1.0, 0: Y_train[Y_train == 1].size / Y_train[Y_train == 0].size}
        else:
            class_weight = {1:Y_train[Y_train == 0].size / Y_train[Y_train == 1].size, 0: 1.0}
        print("class_weight", class_weight)
    else:
        class_weight = None

    pred=siamese_model.predict([X_test1, X_test2])
    predclass=np.argmax(pred, axis=1)
    acc = accuracy_score(Y_test, predclass)
    print(acc)
    f1 = f1_score(Y_test, predclass)
    print(f1)
    cm = confusion_matrix(Y_test, predclass)
    print(cm)
    history = siamese_model.fit(x=[X_train1, X_train2],
                                y=to_categorical(Y_train),
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                validation_data=([X_test1, X_test2], to_categorical(Y_test)),
                                #validation_split=0.025,
                                class_weight=class_weight,
                                callbacks=my_calls)
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
    pred = siamese_model.predict(x=[X_test1, X_test2])
    with open("preds.txt", 'w') as fid:
        fid.write("\n".join(map(str, pred)))
    predclass=np.argmax(pred, axis=1)
    print("acc:", accuracy_score(Y_test, predclass))
    print("f1: ", f1_score(Y_test, predclass))
    print("confusion matrix")
    print(confusion_matrix(Y_test, predclass))

    with open("he_approach_per_epoch_metrics.p", "wb") as fid:
        pickle.dump([history.history, per_epoch_preds.acchis, per_epoch_preds.f1his, per_epoch_preds.cmhis, args], fid)

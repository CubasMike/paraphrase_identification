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

print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
##################
### PARAMETERS ###
##################
# TODO: Implement the parameters as input from command line
version = "20180427"
# myprep, kimprep
pp_name = "kimprep"
lower_opt = "nolower"
emb_opt = "embcap"

reverse_train=False
padding = True
# We later add the full size of sequences
# Note: filter sizes should be different due to naming of the layers
#filters_ga = [(None, None), (1,300), (2,300), (3, 300)]
# NOTE: if there is a filter with None and the number of filters is different
#       to the embedding dim they program will crash
holistic_fnum = 100
filters_ga = [(1,holistic_fnum), (2,holistic_fnum), (3,holistic_fnum)]
perdim_fnum = 20
filters_gb = [(1,perdim_fnum), (2,perdim_fnum), (3,perdim_fnum)]
##################
epochs = 3
batch_size = 32
trainable_embeddings = False
hidden_units_merger = 50
regularization_param = 1e-4
gtypes = ["group_a_only", "group_b_only", "both"]
atypes = ["algo1_only", "algo2_only", "both"]
grouptype = gtypes[0]
algtype = atypes[0]
model = mpcnn_model.he_model_siamese
use_cos = True
use_euc = False#True
use_abs = False#True
autoneg = 0

poolings_ga = [Lambda(lambda x: K.max(x, axis=1), name="ga_maxpool")]
#poolings_ga = [Lambda(lambda x: K.max(x, axis=1), name="ga_maxpool"), Lambda(lambda x: K.min(x, axis=1), name="ga_minpool")]
#poolings_ga = [Lambda(lambda x: K.max(x, axis=1), name="ga_maxpool"), Lambda(lambda x: K.min(x, axis=1), name="ga_minpool"), Lambda(lambda x: K.sum(x, axis=1)/K.sum(K.cast(K.not_equal(x, 0), "float32"), axis=1) , name="ga_avgpool")]

poolings_gb = [Lambda(lambda x: K.max(x, axis=1), name="gb_maxpool")]
#poolings_gb = [Lambda(lambda x: K.max(x, axis=1), name="gb_maxpool"), Lambda(lambda x: K.min(x, axis=1), name="gb_minpool")]
######################
### END PARAMETERS ###
######################

# Loading embeddings matrix
# word2vec, glove, paragram
emb_name = "glove"
emb_fn = "msrpc_{}_{}_{}_{}_{}.pickle".format(pp_name, lower_opt, emb_name, emb_opt, version)
[embeddings_matrix1, unknown_words1] = pickle.load(open("./data/"+emb_fn, 'rb'))
vocab_size = embeddings_matrix1.shape[0]
print("Vocab size: {}".format(vocab_size))
embeddings_dim1 = embeddings_matrix1.shape[1]
print("Embeddings dim 1 (pretrained GloVe): {}  unknown tokens: {}".format(embeddings_dim1, len(unknown_words1)))

emb_name = "word2vec"
emb_fn = "msrpc_{}_{}_{}_{}_{}.pickle".format(pp_name, lower_opt, emb_name, emb_opt, version)
[embeddings_matrix2, unknown_words2] = pickle.load(open("./data/"+emb_fn, 'rb'))
embeddings_dim2 = embeddings_matrix2.shape[1]
print("Embeddings dim 2 (pretrained word2vec): {}  unknown tokens: {}".format(embeddings_dim2, len(unknown_words2)))

emb_name = "paragram25"
emb_fn = "msrpc_{}_{}_{}_{}_{}.pickle".format(pp_name, lower_opt, emb_name, emb_opt, version)
[embeddings_matrix3, unknown_words3] = pickle.load(open("./data/"+emb_fn, 'rb'))
embeddings_dim3 = embeddings_matrix3.shape[1]
print("Embeddings dim 3 (pretrained PARAGRAM): {}  unknown tokens: {}".format(embeddings_dim3, len(unknown_words3)))
assert ((embeddings_matrix1.shape[0] == embeddings_matrix2.shape[0]) and
         (embeddings_matrix1.shape[0] == embeddings_matrix3.shape[0])),\
                 "Embedding matrices were not computed with the same vocabulary"

print("Final embeddings dim:", embeddings_dim1+embeddings_dim2+embeddings_dim3)

# Generating dataset from parsed MSRPC
(index_to_word, word_to_index,
 X_train1, X_train2, Y_train,
 X_test1, X_test2, Y_test) = utils.generate_dataset(pp_name, lower_opt, version,
                                      max_seq_length=-1, reverse_train_pairs=reverse_train,
                                      padding=padding, autoneg=autoneg)

# Decides if we pad all sentences to the same lenght
if padding:
    max_seq_length = X_train1.shape[1]
    print("Max seq length:", max_seq_length)
    print("X_train:", X_train1.shape)
    print("Y_train:", Y_train.shape)
    print("X_test:", X_test1.shape)
    print("Y_test:", Y_test.shape)
    # Adding filters with size same as the entire sentence
    #filters_ga.append((max_seq_length, filters_ga[-1][1]))
    #filters_gb.append((max_seq_length, filters_gb[-1][1]))
else:
    # NOTE: This is not working yet
    max_seq_length = None
    print("Using sequences of different lengths")
    print("X_train:", len(X_train1))
    print("Y_train:", len(Y_train))
    print("X_test:", len(X_test1))
    print("Y_test:", len(Y_test))


print("Filters ga {}".format(filters_ga))
print("Filters gb {}".format(filters_gb))

# Selecting the model
print("Generating model...",)
siamese_model, encoder = model((max_seq_length, ), filters_ga, filters_gb,
                      [embeddings_dim1], [embeddings_matrix1],
                      #[embeddings_dim1, embeddings_dim2, embeddings_dim3],
                      #[embeddings_matrix1, embeddings_matrix2, embeddings_matrix3],
                      #[embeddings_dim1, embeddings_dim3],
                      #[embeddings_matrix1, embeddings_matrix3],
                      word_to_index, max_seq_length, regularization_param,
                      hidden_units_merger, trainable_embeddings,
                      grouptype, algtype, poolings_ga, poolings_gb,
                      use_cos, use_euc, use_abs)
print("Done!")

# Printing summaries
#encoder.summary()
#siamese_model.summary()

# Compiling model
print("Compiling model ...",)
optimizer = optimizers.Adam(lr=0.001)#, clipnorm=1.)
#optimizer = optimizers.SGD(lr=0.001)
losses = ["categorical_crossentropy", "categorical_hinge", "kullback_leibler_divergence"]
my_loss=losses[0]
siamese_model.compile(optimizer=optimizer,
        loss=my_loss,
        metrics=["accuracy"])
print("Done!")


def generator_for_predict(X1, X2, batch_size=1):
    #for i in range(0, len(X1), batch_size):
    for i in range(0, len(X1)):
        s1 = np.expand_dims(X1[i], axis=0)
        s2 = np.expand_dims(X2[i], axis=0)
        #print(s1)
        #print(s2)
        yield [s1, s2]

def generator_for_fit(X1, X2, Y, batch_size=1):
    #for i in range(0, len(X1), batch_size):
    steps = 0
    #for i in range(0, len(X1)):
    while True:
        steps += 1
        i = (steps-1)%len(X1)
        s1 = np.expand_dims(X1[i], axis=0)
        s2 = np.expand_dims(X2[i], axis=0)
        label = np.expand_dims(Y[i], axis=0)
        #print(s1)
        #print(s2)
        #print(label)
        yield [s1, s2], label

# Callback to store prediction scores for each epoch
class prediction_history(Callback):
    def __init__(self):
        self.acchis = []
        self.f1his = []
        self.cmhis = []
    def on_epoch_end(self, epoch, logs={}):
        if padding:
            pred=self.model.predict([X_test1, X_test2])
        else:
            pred=self.model.predict_generator(generator_for_predict(X_test1, X_test2), steps=len(X_test1))
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
print("Training model ...",)
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

use_class_weight = False
if use_class_weight:
    if Y_train[Y_train == 1].size > Y_train[Y_train == 0].size:
        class_weight = {1:1.0, 0: Y_train[Y_train == 1].size / Y_train[Y_train == 0].size}
    else:
        class_weight = {1:Y_train[Y_train == 0].size / Y_train[Y_train == 1].size, 0: 1.0}
    print("class_weight", class_weight)
else:
    class_weight = None

if padding:
    pred=siamese_model.predict([X_test1, X_test2])
else:
    pred=siamese_model.predict_generator(generator_for_predict(X_test1, X_test2), steps=len(X_test1))
predclass=np.argmax(pred, axis=1)
acc = accuracy_score(Y_test, predclass)
print(acc)
f1 = f1_score(Y_test, predclass)
print(f1)
cm = confusion_matrix(Y_test, predclass)
print(cm)
if padding:
    history = siamese_model.fit(x=[X_train1, X_train2],
                                y=to_categorical(Y_train),
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=([X_test1, X_test2], to_categorical(Y_test)),
                                #validation_split=0.025,
                                class_weight=class_weight,
                                callbacks=my_calls)
    print("Done!")
else:
    # Using fit_generator
    print("Training model ... batch-by-batch")
    print(type(X_train1), X_train1.shape)
    print(type(X_train2), X_train2.shape)
    print(type(Y_train), Y_train.shape)
    print(np.expand_dims(X_train1[0], axis=0))
    print(np.expand_dims(X_train2[0], axis=0))
    print(np.expand_dims(to_categorical(Y_train)[0], axis=0))
    print("Steps per epoch:", len(X_train1))
    batch_size = 1
    history = siamese_model.fit_generator(generator_for_fit(X_train1, X_train2, to_categorical(Y_train)),
                                          epochs=epochs,
                                          steps_per_epoch=len(X_train1),
                                          class_weight=class_weight,
                                          callbacks=my_calls)

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
if padding:
    pred = siamese_model.predict(x=[X_test1, X_test2])
else:
    pred = siamese_model.predict_generator(generator_for_predict(X_test1, X_test2), steps=len(X_test1))

with open("preds.txt", 'w') as fid:
    fid.write("\n".join(map(str, pred)))
predclass=np.argmax(pred, axis=1)
print("acc:", accuracy_score(Y_test, predclass))
print("f1: ", f1_score(Y_test, predclass))
print("confusion matrix")
print(confusion_matrix(Y_test, predclass))

with open("per_epoch_metrics.p", "wb") as fid:
    pickle.dump([history.history, per_epoch_preds.acchis, per_epoch_preds.f1his, per_epoch_preds.cmhis], fid)



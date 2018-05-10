import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
from keras import optimizers

from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import pickle
import lexdecomp_model
import utils
import datetime

###################
# DATA PARAMETERS #
###################
version = "20180427"
# myprep, kimprep
pp_name = "kimprep"
lower_opt = "nolower"
emb_opt = "embcap"
# word2vec, glove, paragram
emb_name = "word2vec"
# Duplicate training by switching pairs
reverse_train=False
# Randomly generate negative samples
autoneg = 0
#######################
# END DATA PARAMETERS #
#######################

# Generating dataset from parsed MSRPC
(index_to_word, word_to_index,
 X_train1, X_train2, Y_train,
 X_test1, X_test2, Y_test) = utils.generate_dataset(pp_name, lower_opt, version,
                                      max_seq_length=-1,
                                      reverse_train_pairs=reverse_train,
                                      padding=True,
                                      autoneg=autoneg)
#max_seq_length = 39
max_seq_length = X_train1.shape[1]
print("Max seq length:", max_seq_length)
print("X_train:", X_train1.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test1.shape)
print("Y_test:", Y_test.shape)

# Loading embeddings matrix
emb_fn = "msrpc_{}_{}_{}_{}_{}.pickle".format(pp_name, lower_opt, emb_name, emb_opt, version)
[embedding_matrix, unknown_words] = pickle.load(open("./data/"+emb_fn, 'rb'))
embeddings_dim = embedding_matrix.shape[1]
print("Embeddings dim:", embeddings_dim)


####################
# MODEL PARAMETERS #
####################
epochs = 30
batch_size = 64
window = 3
# method = linear or orthogonal
method = "orthogonal"
filters = [(1,500), (2,500), (3,500)]
use_class_weight = False
############################
### END MODEL PARAMETERS ###
############################

# Transforming train data from sequence of indeces to
# sequence of embeddings
# shape input: (samples, max_seq_length)
#      output: (samples, max_seq_length, embeddings_dim)
X_train1 = lexdecomp_model.transform_data(X_train1, embedding_matrix)
X_train2 = lexdecomp_model.transform_data(X_train2, embedding_matrix)
X_test1 = lexdecomp_model.transform_data(X_test1, embedding_matrix)
X_test2 = lexdecomp_model.transform_data(X_test2, embedding_matrix)

# Decomposing train and test data
# shape output: (samples, max_seq_length, embeddings_dim, 2)
print("Decomposing training data")
X_train1, X_train2 = lexdecomp_model.decompose_data(X_train1, X_train2, window, method)
print("Decomposing test data")
X_test1, X_test2 = lexdecomp_model.decompose_data(X_test1, X_test2, window, method)
print("Decomposed data")
print("X_train:", X_train1.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test1.shape)
print("Y_test:", Y_test.shape)

# Selecting the model
model = lexdecomp_model.lexdecomp_model((max_seq_length, embeddings_dim, 2),
                                        embeddings_dim, max_seq_length, filters)
# Printing summaries
model.summary(line_length=100)

# Compiling model
model.compile(optimizer="Adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Training model
# Defining class weights for unbalanced datasets
if use_class_weight:
    if Y_train[Y_train == 1].size > Y_train[Y_train == 0].size:
        class_weight = {1:1.0, 0: Y_train[Y_train == 1].size / Y_train[Y_train == 0].size}
    else:
        class_weight = {1:Y_train[Y_train == 0].size / Y_train[Y_train == 1].size, 0: 1.0}
    print("class_weight", class_weight)
else:
    class_weight = None

# Callback to store prediction scores for each epoch
class prediction_history(Callback):
    def __init__(self):
        self.acchis = []
        self.f1his = []
        self.cmhis = []
    def on_epoch_end(self, epoch, logs={}):
        pred=self.model.predict([X_test1, X_test2])
        predclass = np.where(pred>0.5, 1, 0).reshape(-1)
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
print("Training model ...")
my_calls = [per_epoch_preds]#None#[es]

history = model.fit(x=[X_train1, X_train2],
                    y=Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    #validation_split=0.1,
                    validation_data=([X_test1, X_test2], Y_test),
                    class_weight=class_weight,
                    callbacks=my_calls)


print("Evaluation (loss, acc)")
loss, acc = model.evaluate(x=[X_test1, X_test2], y=Y_test)
print("loss: {:.4f}   acc: {:.4f}".format(loss, acc))
with open("tmp.p", "wb") as fid:
    pickle.dump(model.history.history, fid)
pred = np.where(model.predict(x=[X_test1, X_test2])>0.5, 1, 0).reshape(-1)
f1 = f1_score(Y_test, pred)
print("f1: {:.4f}".format(f1))
print("confusion matrix")
cf_mat = confusion_matrix(Y_test, pred)
print(cf_mat)
history.history["test_loss"] = loss
history.history["test_acc"] = acc
history.history["f1"] = f1
history.history["cf_mat"] = cf_mat
history.history["pred"] = pred

hdir = "./runs/wang_lexdecomp/"
date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
hfname = hdir + "hist_" + date + "_wang_lexdecopm.p"
with open(hfname, "wb") as fid:
    pickle.dump(history.history, fid)

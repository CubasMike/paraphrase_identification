import pickle
import numpy as np

def convert_to_sequence(texts, word_to_index, padding=False, size_limit=10):
    sequences = {}
    for idx, tokens in texts.items():
        if padding:
            sequences[idx] = np.array([word_to_index[token] for i, token in enumerate(tokens)
                                      if i < size_limit] + [0]*(max(0, size_limit-len(tokens))),
                                      dtype=np.int32)
        else:
            sequences[idx] = np.array([word_to_index[token] for token in tokens],
                                      dtype=np.int32)
    return sequences

def generate_dataset(pp_name, lower_opt, version, max_seq_length=-1,
            reverse_train_pairs=False, padding=True, autoneg=0):
    if padding:
        res = generate_dataset_with_padding(pp_name, lower_opt, version, max_seq_length,
                reverse_train_pairs, autoneg)
    else:
        res = generate_dataset_without_padding(pp_name, lower_opt, version, max_seq_length,
                reverse_train_pairs, autoneg)
    return res


def generate_dataset_with_padding(pp_name, lower_opt, version, max_seq_length=-1,
            reverse_train_pairs=False, autoneg=0):
    parsed_fn = "msrpc_{}_{}_{}.pickle".format(pp_name, lower_opt, version)

    # Loading pre-processed corpus
    [parsed_texts,
     index_to_word,
     word_to_index,
     pairs_train,
     Y_train_list,
     pairs_test,
     Y_test_list] = pickle.load(open("./data/"+parsed_fn, 'rb'))

    # Computing the max_seq_length if not provided
    if max_seq_length < 0:
        max_seq_length = np.max([len(tokens) for idx, tokens in parsed_texts.items()])

    # Transforming list of tokens to sequence of indices
    sequences = convert_to_sequence(parsed_texts, word_to_index,
                                    padding=True, size_limit=max_seq_length)

    # Training Data
    if reverse_train_pairs:
        X_train1 = np.zeros((len(pairs_train)*2+autoneg, max_seq_length), dtype=np.int32)
        X_train2 = np.zeros((len(pairs_train)*2+autoneg, max_seq_length), dtype=np.int32)
        for i, (x1, x2) in enumerate(pairs_train):
            X_train1[i*2,:] = sequences[x1]
            X_train1[i*2+1,:] = sequences[x2]
            X_train2[i*2,:] = sequences[x2]
            X_train2[i*2+1,:] = sequences[x1]
        Y_train = np.array([Y_train_list[i//2] for i in range(len(Y_train_list)*2)]+[0]*autoneg, dtype=np.int32)
    else:
        X_train1 = np.zeros((len(pairs_train)+autoneg, max_seq_length), dtype=np.int32)
        X_train2 = np.zeros((len(pairs_train)+autoneg, max_seq_length), dtype=np.int32)
        for i, (x1, x2) in enumerate(pairs_train):
            X_train1[i,:] = sequences[x1]
            X_train2[i,:] = sequences[x2]
        Y_train = np.array(Y_train_list+[0]*autoneg, dtype=np.int32)

    # Adding automatically generated negative samples
    # from sentences in positive samples
    left, right = zip(*[tup for tup, _class in zip(pairs_train, Y_train_list) if _class==1])
    pos_ids = np.array(list(set(left+right)), dtype=np.int32)
    selected_pos_ids = np.random.choice(pos_ids, size=autoneg)
    pairs_train_set = set(pairs_train)
    pairs_test_set = set(pairs_test)
    all_ids = np.array(list(parsed_texts.keys()), dtype=np.int32)
    starting_i = len(pairs_train)*2 if reverse_train_pairs else len(pairs_train)
    for i, pos_id in enumerate(selected_pos_ids, start=starting_i):
        while True:
            paired_id = np.random.choice(all_ids)
            # Check it is not in test set too
            if ((pos_id, paired_id) not in pairs_train_set and
                   (paired_id, pos_id) not in pairs_train_set and
                   (pos_id, paired_id) not in pairs_test_set and
                   (paired_id, pos_id) not in pairs_test_set):
                X_train1[i,:] = sequences[pos_id]
                X_train2[i,:] = sequences[paired_id]
                break
            else:
                print("Ignoring randomly generated sample that already exists")

    # Test Data
    X_test1 = np.zeros((len(pairs_test), max_seq_length), dtype=np.int32)
    X_test2 = np.zeros((len(pairs_test), max_seq_length), dtype=np.int32)
    for i, (x1, x2) in enumerate(pairs_test):
        X_test1[i,:] = sequences[x1]
        X_test2[i,:] = sequences[x2]
    Y_test = np.array(Y_test_list, dtype=np.int32)

    return index_to_word, word_to_index, X_train1, X_train2, Y_train, X_test1, X_test2, Y_test


def generate_dataset_without_padding(pp_name, lower_opt, version, max_seq_length=-1,
            reverse_train_pairs=False, autoneg=0):
    parsed_fn = "msrpc_{}_{}_{}.pickle".format(pp_name, lower_opt, version)

    # Loading pre-processed corpus
    [parsed_texts,
     index_to_word,
     word_to_index,
     pairs_train,
     Y_train_list,
     pairs_test,
     Y_test_list] = pickle.load(open("./data/"+parsed_fn, 'rb'))

    # Computing the max_seq_length if not provided
    if max_seq_length < 0:
        max_seq_length = np.max([len(tokens) for idx, tokens in parsed_texts.items()])

    # Transforming list of tokens to sequence of indices
    sequences = convert_to_sequence(parsed_texts, word_to_index,
                                    padding=False, size_limit=max_seq_length)

    # Training Data
    if reverse_train_pairs:
        X_train1 = []
        X_train2 = []
        for i, (x1, x2) in enumerate(pairs_train):
            X_train1.append(np.array(sequences[x1], dtype=np.int32))
            X_train1.append(np.array(sequences[x2], dtype=np.int32))
            X_train2.append(np.array(sequences[x2], dtype=np.int32))
            X_train2.append(np.array(sequences[x1], dtype=np.int32))
        Y_train = np.array([Y_train_list[i//2] for i in range(len(Y_train_list)*2)]+[0]*autoneg, dtype=np.int32)
    else:
        X_train1 = []
        X_train2 = []
        for i, (x1, x2) in enumerate(pairs_train):
            X_train1.append(np.array(sequences[x1], dtype=np.int32))
            X_train2.append(np.array(sequences[x2], dtype=np.int32))
        Y_train = np.array(Y_train_list+[0]*autoneg, dtype=np.int32)

    # Adding automatically generated negative samples
    # from sentences in positive samples
    left, right = zip(*[tup for tup, _class in zip(pairs_train, Y_train_list) if _class==1])
    pos_ids = np.array(list(set(left+right)), dtype=np.int32)
    selected_pos_ids = np.random.choice(pos_ids, size=autoneg)
    pairs_train_set = set(pairs_train)
    pairs_test_set = set(pairs_test)
    all_ids = np.array(list(parsed_texts.keys()), dtype=np.int32)
    starting_i = len(pairs_train)*2 if reverse_train_pairs else len(pairs_train)
    for i, pos_id in enumerate(selected_pos_ids, start=starting_i):
        while True:
            paired_id = np.random.choice(all_ids)
            # Check it is not in test set too
            if ((pos_id, paired_id) not in pairs_train_set and
                   (paired_id, pos_id) not in pairs_train_set and
                   (pos_id, paired_id) not in pairs_test_set and
                   (paired_id, pos_id) not in pairs_test_set):
                X_train1.append(np.array(sequences[pos_id], dtype=np.int32))
                X_train2.append(np.array(sequences[paired_id], dtype=np.int32))
                break
            else:
                print("Ignoring randomly generated sample that already exists")

    # Test Data
    X_test1 = []
    X_test2 = []
    for i, (x1, x2) in enumerate(pairs_test):
        X_test1.append(np.array(sequences[x1], dtype=np.int32))
        X_test2.append(np.array(sequences[x2], dtype=np.int32))
    Y_test = np.array(Y_test_list, dtype=np.int32)

    return (index_to_word, word_to_index,
            np.array(X_train1), np.array(X_train2), Y_train,
            np.array(X_test1), np.array(X_test2), Y_test)

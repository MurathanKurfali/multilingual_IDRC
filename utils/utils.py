import random
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score




def shuffle_set(arg1, arg2, y):
    all = list(zip(arg1, arg2, y))
    random.shuffle(all)
    arg1_sentences, arg2_sentences, y = zip(*all)
    arg1_sentences = np.array(arg1_sentences)
    arg2_sentences = np.array(arg2_sentences)
    y = np.array(y)

    return arg1_sentences, arg2_sentences, y


# read laser embeddings
def read_laser_embeddings(argument_embed_dir, label_dir, lang,  device, sense,  set_type):
    dim = 1024
    argument_embed_file = argument_embed_dir + "%s_implicit_%s"
    label_file = label_dir + "%s_implicit_sense.txt"
    if len(lang) != 2:
        lang = lang+"_"+set_type

    arg1_sentences = np.fromfile(argument_embed_file % (lang, "arg1"), dtype=np.float32, count=-1)
    arg1_sentences.resize(arg1_sentences.shape[0] // dim, dim)

    arg2_sentences = np.fromfile(argument_embed_file % (lang, "arg2"), dtype=np.float32, count=-1)
    arg2_sentences.resize(arg2_sentences.shape[0] // dim, dim)

    labels = [s.replace("\n", "").split("\t")[0] for s in open(label_file % lang, "r").readlines()]
    labels = [sense if s == sense else "xnon-" + sense for s in labels]

    # perform upsampling
    if set_type != "test":
        # upsampling (Expansion vs others)
        if labels.count(sense) > labels.count("xnon-" + sense):
            negative_example_indices = [i for i in range(len(labels)) if labels[i] != sense]
            diff = (labels.count(sense) - labels.count("xnon-" + sense))
            random_extra = [random.choice(negative_example_indices) for _ in range(diff)]
            extra_arg1 = arg1_sentences[random_extra]
            extra_arg2 = arg2_sentences[random_extra]
            extra_labels = ["xnon-" + sense] * diff

            arg1_sentences = np.concatenate((arg1_sentences, extra_arg1), 0)
            arg2_sentences = np.concatenate((arg2_sentences, extra_arg2), 0)
            labels = labels + extra_labels

        # upsampling (Temp, Comp, Cont vs others)
        elif labels.count("xnon-" + sense) > labels.count(sense):
            positive_example_indices = [i for i in range(len(labels)) if labels[i] == sense]
            diff = (labels.count("xnon-" + sense) - labels.count(sense))
            random_extra = [random.choice(positive_example_indices) for _ in range(diff)]
            extra_arg1 = arg1_sentences[random_extra]
            extra_arg2 = arg2_sentences[random_extra]
            extra_labels = [sense] * diff

            arg1_sentences = np.concatenate((arg1_sentences, extra_arg1), 0)
            arg2_sentences = np.concatenate((arg2_sentences, extra_arg2), 0)
            labels = labels + extra_labels

        assert labels.count(sense) == labels.count("xnon-" + sense)

    y = np.array([[sense, "xnon-" + sense].index(l) for l in labels], dtype=np.float32)
    arg1_sentences, arg2_sentences, y = shuffle_set(arg1_sentences, arg2_sentences, y)

    ## convert to pytorch variable
    arg1_sentences = torch.autograd.Variable(torch.from_numpy(arg1_sentences), requires_grad=False).to(device)
    arg2_sentences = torch.autograd.Variable(torch.from_numpy(arg2_sentences), requires_grad=False).to(device)
    y = torch.autograd.Variable(torch.FloatTensor(y), requires_grad=False).to(device)

    return arg1_sentences, arg2_sentences, y


def prepare_input(embeddings_dir, label_dir, lang, sense, device, type):
    arg1_embed = []
    arg2_embed = []
    label_list = []

    for i in range(len(embeddings_dir)):
        arg1, arg2, label = read_laser_embeddings(embeddings_dir[i], label_dir[i], lang[i], device, sense=sense, set_type=type)
        arg1_embed.append(arg1)
        arg2_embed.append(arg2)
        label_list.append(label)

    arg1_torch = torch.cat(arg1_embed)
    arg2_torch = torch.cat(arg2_embed)
    label_torch = torch.cat(label_list)

    return arg1_torch, arg2_torch, label_torch


## calc accuracy
def calc_accuracy(Y, y_pred, verbose=False):

    y_pred = torch.argmax(y_pred, 1, keepdim=False)
    predictions = y_pred.cpu().data.numpy()
    targets = Y.cpu().data.numpy()
    average = "binary"

    most_common_label = Counter(targets).most_common(1)
    dummy_model_pred = np.repeat(most_common_label[0][0], len(predictions))

    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, pos_label=0, average=average)
    prec = precision_score(targets, predictions, pos_label=0, average=average)
    rec = recall_score(targets, predictions, pos_label=0, average=average)

    most_common_baseline_acc = accuracy_score(targets, dummy_model_pred)
    most_common_baseline_f1 = f1_score(targets, dummy_model_pred, pos_label=0, average=average)

    if verbose:
        print("acc: {:.3f} prec: {:.3f} recall: {:.3f} f1: {:.3f}".format(acc, prec, rec, f1))
        print("baseline acc: {:.3f}  f1: {:.3f}".format(most_common_baseline_acc, most_common_baseline_f1))

    return acc, f1, prec, rec, most_common_baseline_acc, most_common_baseline_f1


if __name__ == "__main__":

    read_laser_embeddings("../data/embed", "../data/text", "de", "Temporal")
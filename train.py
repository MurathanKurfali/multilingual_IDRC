import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from model import relation_classifier
from utils.utils import calc_accuracy, prepare_input
import argparse, os


def train(embeddings_dir, label_dir, out_dir, training_langs, sense_list, epoch_num, batch_size, early_stopping_threshold, device, verbose = False):
    for sense in sense_list:
        print("~~~ Training the \" %s vs Others \" classifier" % (sense))
        best_score = -1
        score_list = []
        out_file = out_dir + "/%s_%s_model" % ("_".join(training_langs), sense)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Prepare Data
        dev_arg1, dev_arg2, dev_label = prepare_input(embeddings_dir, label_dir, training_langs, sense, device, type="dev")

        # model
        input_dim = dev_arg1.shape[1]*5  # *5 due to the dissent model
        network = relation_classifier(input_dim).to(device)
        optimizer = optim.Adagrad(network.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        dev_label = dev_label.long()
        if verbose:
            print(network, "\n\n")


        # training loop
        for epoch in range(epoch_num):
            train_arg1, train_arg2, train_label = prepare_input(embeddings_dir, label_dir, training_langs, sense, device, type="training")
            train_label = train_label.long()

            for batch in range(int(len(train_label)/batch_size)+1):
                arg1 = train_arg1[batch*batch_size:min((batch+1)*batch_size, len(train_arg1))]
                arg2 = train_arg1[batch*batch_size:min((batch+1)*batch_size, len(train_arg2))]
                batch_labels = train_label[batch*batch_size:min((batch+1)*batch_size, len(train_label))]
                optimizer.zero_grad()
                y_pred = network(arg1, arg2)
                output = loss_fn(y_pred, batch_labels)

                output.backward()
                optimizer.step()

            ## compute validation loss
            valid_predictions = network(dev_arg1, dev_arg2)
            acc, f1, prec, rec, base_acc, base_f1 = calc_accuracy(dev_label, valid_predictions, verbose=False)
            if verbose:
                print("Epoch {} Validation acc: {:.7f}  f1: {:.7f}".format(epoch, acc, f1), sep=" ",  flush=True)
            elif epoch % 10 == 0:
                print("Epoch {} Validation acc: {:.7f}  f1: {:.7f}".format(epoch, acc, f1), sep=" ",  flush=True)

            ## early stopping
            if f1 > best_score:
                torch.save(network.state_dict(), out_file)
                if verbose:
                    print("model saved at Epoch", epoch)
                best_score = f1
                score_list = []
            else:
                score_list.append(f1)
            if len(score_list) > early_stopping_threshold:
                print("Early Stopping after {} iteration".format( epoch))
                print("the highest F-score achieved on the validation data: %.4f" % best_score)
                print("The model is saved to", out_file)
                print("--------------------")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multilingual IDRC. By default, this script will train 4 \" one vs all \" classifiers on PDTB3 training data.')
    parser.add_argument('--embed_dir', type=str, required=True, help="The directory of laser embeddings")
    parser.add_argument('--label_dir', type=str, required=True, help="The directory of labels ")
    parser.add_argument('--out_dir', type=str, required=True, help="The directory where the models will be saved")

    parser.add_argument('--training_sets', nargs="+", default=["pdtb3"], required=True,
                        help="Training sets to train the model, options are: \n pdtb3 (default) \npdtb2: \ntdb: Turkish Discourse Bank")
    parser.add_argument('--senses', nargs="+", default=["Comparison", "Contingency", "Expansion", "Temporal"],
                        help="List of senses to train the classifier")

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--early', type=int, default=30)

    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--verbose', "-v", action="store_true")

    args = parser.parse_args()

    epoch = args.epoch
    training_sets = args.training_sets
    out_dir = args.out_dir
    sense_list = args.senses

    batch = args.batch
    early_stopping_threshold = args.early
    verbose = args.verbose

    if args.gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    embeddings_dir = [args.embed_dir  for tr_set in training_sets]
    label_dir = [args.label_dir  for lang in training_sets]
    print("the model will be trained on {} using {} for the following senses: \n- {}\n".format(training_sets, device.upper(), " \n- ".join(sense_list)))
    print("--------------------")
    train(embeddings_dir, label_dir, out_dir, training_sets,  sense_list, epoch, batch, early_stopping_threshold, device,  verbose)

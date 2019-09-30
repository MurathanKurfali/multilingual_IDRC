import torch
from model import relation_classifier
from utils.utils import calc_accuracy, read_laser_embeddings
import argparse, sys, os
from collections import defaultdict
import numpy as np


output_dim = 2


def eval_model(embedding_dir, label_dir, sense, model_dir, target_languages, device):
    result_dict = defaultdict(float)

    net = relation_classifier(1024 * 5, output_dim).to(device)
    net.load_state_dict(torch.load(model_dir, map_location=device))

    for test_set in target_languages:
        if test_set == "ted":
            languages = ["tr", "de", "pl", "ru", "pt", "en", "lt"]
            for lang in languages:
                ted_test_arg1, ted_test_arg2, ted_test_target = read_laser_embeddings(embedding_dir, label_dir, lang, device, sense=sense, set_type="test")
                test_pred = net(ted_test_arg1, ted_test_arg2)  # predict
                acc, f1, prec, rec, base_acc, base_f1 = calc_accuracy(ted_test_target, test_pred, verbose=False)
                result_dict[lang] = (f1)
        else:
            test_arg1, test_arg2, test_target = read_laser_embeddings(embedding_dir, label_dir, test_set, device, sense=sense,  set_type="test")
            test_pred = net(test_arg1, test_arg2)  # predict
            acc, f1, prec, rec, base_acc, base_f1 = calc_accuracy(test_target, test_pred, verbose=False)
            result_dict[test_set] = (f1)

    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Multilingual IDRC model')
    parser.add_argument('--model_file', type=str, help="Test a specific model")
    parser.add_argument('--model_dir', type=str, help="Test all the models in the given directory")
    parser.add_argument('--embed_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)

    parser.add_argument('--target', nargs="+", default=["pdtb3"], help="ted = Ted-MDB corpus (default); pdtb2=PDTB2 test set; pdtb3= PDTB3 test set")
    parser.add_argument('--gpu', action="store_true")

    args = parser.parse_args()

    if args.model_dir is None and args.model_file is None:
        print("No model file is provided!")
        sys.exit()

    model_dir = args.model_dir
    model_file = args.model_file

    embeddings_dir = args.embed_dir
    label_dir = args.label_dir
    target_languages = args.target
    result_dict = defaultdict(lambda: defaultdict(list))

    if args.gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    if model_dir is not None:
        model_files = {filename.split("_")[1]: os.path.join(model_dir, filename) for filename in os.listdir(model_dir)}
    else:
        model_files = {model_file.split("_")[-2]: model_file}

    for sense, model in model_files.items():
        print("Testing %s with using model %s on %s" % (sense, model, device.upper()))
        x = eval_model(embeddings_dir, label_dir, sense, model, target_languages, device)
        for k, v in x.items():
            result_dict[sense][k].append(v)

    for sense, res_dict in result_dict.items():
        print(sense)
        for lang, result in res_dict.items():
            print("%s f-score= %.3f" % (lang.upper(), np.average(result)))

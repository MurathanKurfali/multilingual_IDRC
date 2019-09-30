import os, sys
import re
from collections import defaultdict
import numpy as np
from gensim.parsing.preprocessing import strip_punctuation


class pdtb_reader:

    def __init__(self, ann_dir, raw_dir, is_pdtb3):
        self.ann_dir = ann_dir
        self.raw_dir = raw_dir
        self.is_pdtb3 = is_pdtb3
        self.rel_with_senses = ["Explicit", "Implicit", "AltLex"]

        self.file_list = sorted([x[0].split("/")[-1] + "/" + y for x in os.walk(ann_dir) for y in x[2] if ".xml" not in y])
        self.relations = defaultdict(lambda: defaultdict(list))
        for f in self.file_list:
            self.extract_anno(f)

    def extract_anno(self, file):
        chapter = file.split("/")[0]
        annotations = open(self.ann_dir + file, "r", encoding="utf8").readlines()
        raw = open(self.raw_dir + file, "r", encoding="utf8").read()
        for anno in annotations:
            if "Rejected" not in anno:
                self.anno = anno
                anno = anno.split("|")
                type = anno[0]
                conn = anno[1]
                if self.is_pdtb3:
                    arg1 = self.extract_arg(raw, anno[14])
                    arg2 = self.extract_arg(raw, anno[20])
                else:
                    arg1 = self.extract_arg(raw, anno[22])
                    arg2 = self.extract_arg(raw, anno[32])
                if type in self.rel_with_senses:
                    if "Hypophora" in anno[8]:
                        continue
                    if self.is_pdtb3:
                        sense = "\t".join([anno[i].split(".")[0] for i in [8, 9] if anno[i] != ""])
                    else:
                        sense = "\t".join([anno[i].split(".")[0] for i in [11, 12, 13, 14] if anno[i] != ""])
                    sense = self.clear_sense(sense)
                    if conn != "":
                        self.relations[chapter][type].append((arg1, arg2, sense, conn))
                    else:
                        self.relations[chapter][type].append((arg1, arg2, sense))
                else:
                    self.relations[chapter][type].append((arg1, arg2))

    # to clear the sense annotations in the polish files
    def clear_sense(self, senses):
        sense_list = senses.split("\t")
        parenthesis_count = np.sum(["(" in s for s in sense_list])
        if parenthesis_count > 0:
            senses = "\t".join([s[s.find("(") + 1:s.find(")")] for s in sense_list])
        return senses

    def extract_arg(self, raw, indices):
        if ";" not in indices:
            try:
                start, end = map(int, indices.split(".."))
            except:
                print("invalid annotation", self.anno)
                sys.exit()
            arg = raw[start:end]
            arg = strip_punctuation(arg).replace("\n", "").lower()
            arg = re.sub('\s{2,}', ' ', arg).strip()
            return arg.split(" ")
        else:
            arg = ""
            for i in indices.split(";"):
                try:
                    start, end = map(int, i.split(".."))
                except:
                    continue
                arg += raw[start:end] + " "
            arg = strip_punctuation(arg[:-1]).replace("\n", "").lower()
            arg = re.sub('\s{2,}', ' ', arg).strip()

            return arg.split(" ")

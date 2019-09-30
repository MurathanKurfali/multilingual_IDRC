#!/usr/bin/env bash
## GENERATE LASER EMBEDDINGS
raw_sentence_dir="data/text"
sentence_embed_dir="data/embed"
mkdir $sentence_embed_dir
arg_array=(arg1 arg2)

lang_array=()

for type in "$@"
do
    if [ $type == "pdtb3" ]
    then
        lang_array+=(pdtb3_dev pdtb3_test pdtb3_training)
    fi
    if [ $type == "pdtb2" ]
    then
        lang_array+=(pdtb2_dev pdtb2_test pdtb2_training)
    fi
    if [ $type == "ted" ]
    then
        lang_array+=(de en pt ru tr pl lt)
    fi
done

# if no valid argument is provided!
if [ ${#lang_array[@]} -eq 0 ]; then
    exit 1
fi

for lang in "${lang_array[@]}"; do
    echo "$lang"
    for arg in "${arg_array[@]}"; do
        echo "$arg";
        "${LASER}/tasks/embed/embed.sh" "${raw_sentence_dir}/${lang}_implicit_${arg}.txt" $lang "${sentence_embed_dir}/${lang}_implicit_${arg}"
    done
done

from pdtb_reader import pdtb_reader
import argparse
import os

language_dict = {"Polish": "pl", "English": "en", "German": "de", "Russian": "ru", "Portuguese": "pt", "Turkish": "tr", "lithuanian": "lt"}


def extract_ted_mdb(dir):
    ann_dir = dir + "/%s/ann/"
    raw_dir = dir + "/%s/raw/"
    output_dir = "data/text/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for lang, lang_code in language_dict.items():
        #print(lang)
        reader = pdtb_reader(ann_dir % lang, raw_dir % lang, is_pdtb3=True)
        out_file = output_dir + lang_code + "_implicit_%s.txt"
        with open(out_file % "arg1", "w", encoding="utf8") as arg1_file:
            with open(out_file % "arg2", "w", encoding="utf8") as arg2_file:
                with open(out_file % "sense", "w", encoding="utf8") as sense_file:
                    for annotation in reader.relations["01"]["Implicit"]:
                        arg1_file.write(" ".join(annotation[0]) + "\n")
                        arg2_file.write(" ".join(annotation[1]) + "\n")
                        sense_file.write(annotation[2] + "\n")

        arg1_file.close()
        arg2_file.close()
        sense_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='Extract TED-MDB/PDTB relations from annotation files')
    parser.add_argument('--dir', type=str, required=True)

    args = parser.parse_args()
    if not os.path.exists(args.dir):
        print("invalid path!")
    else:
        extract_ted_mdb(args.dir)

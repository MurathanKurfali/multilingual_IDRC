from pdtb_reader import pdtb_reader
import argparse
import os, glob


def extract_pdtb(pdtb_dir, is_pdtb2=False):

    pdtb_ann_dir = os.path.join(pdtb_dir, "ann", "")
    pdtb_raw_dir = os.path.join(pdtb_dir, "raw", "")
    file_name_template = "pdtb2_%s_implicit_%s.txt" if is_pdtb2  else "pdtb3_%s_implicit_%s.txt"
    out_dir = "data/text/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ### remove old files!
    for filename in glob.glob(out_dir + ("pdtb2*" if is_pdtb2  else "pdtb3*")):
        os.remove(filename)

    reader = pdtb_reader(pdtb_ann_dir, pdtb_raw_dir, (not is_pdtb2))
    chapters = reader.relations.keys()

    training_chapters = range(2, 21)
    dev_chapters = list(range(0, 2)) + list(range(23, 25))
    test_chapters = range(21, 23)

    for chapter in chapters:
        if int(chapter) in dev_chapters:
            out_file = out_dir + file_name_template % ("dev", "%s")
            print("dev", chapter)
        elif int(chapter) in test_chapters:
            out_file = out_dir + file_name_template % ("test", "%s")
            print("test", chapter)
        elif int(chapter) in training_chapters:
            out_file = out_dir + file_name_template % ("training", "%s")
            print("training", chapter)

        with open(out_file % "arg1", "a", encoding="utf8") as arg1_file:
            with open(out_file % "arg2", "a", encoding="utf8") as arg2_file:
                with open(out_file % "sense", "a", encoding="utf8") as sense_file:
                    for a in reader.relations[chapter]["Implicit"]:
                        arg1_file.write(" ".join(a[0]) + "\n")
                        arg2_file.write(" ".join(a[1]) + "\n")
                        sense_file.write(a[2] + "\n")

        arg1_file.close()
        arg2_file.close()
        sense_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract PDTB relations from the annotation/raw files')
    parser.add_argument('--dir', type=str, required=True, help="The path of pdtb annotation folder")
    parser.add_argument("--pdtb2", type=bool, default=False, help="whether the annotations are done using PDTB 2 guidelines")

    args = parser.parse_args()
    if not os.path.exists(args.dir):
        print("invalid path!")
    elif not os.path.exists(os.path.join(args.dir, "ann")):
        print("invalid ann path. Please put your ann files under {}/ann".format(args.dir))
    elif not  os.path.exists(os.path.join(args.dir, "raw")):
        print("invalid raw path. Please put your raw files under {}/raw".format(args.dir))
    else:
        extract_pdtb(args.dir, args.pdtb2)
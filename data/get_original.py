import sys
import os
import spacy
import json
import ast
from os import listdir
from bs4 import BeautifulSoup
from tqdm import tqdm

# "/mnt/nfs/scratch1/dongxuzhang/dataset/LDC2011T07/"
gigaword_path = sys.argv[1]
output_dir = sys.argv[2]

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')

gigawords_dirs = ["gigaword_eng_5_d1",
                  "gigaword_eng_5_d2", "gigaword_eng_5_d3"]
source2dir = {"afp": 0, "apw": 0, "cna": 1,
              "ltw": 1, "nyt": 1, "wpb": 1, "xin": 2}

train_dirs = ["data/train/train_-11_10", "data/train/train_-31_30",
              "data/train/train_-101_100", "data/train/train_-301_300"]
relations = ["0", "1", "3", "4", "5", "6", "7"]
# train_files = [f"{d}_{r}.txt" for d in train_dirs for r in relations] + [
#    "data/train/train_negatives.txt"]
train_files = []

val_files = ["data/val/0.txt", "data/val/1.txt",
             "data/val/3.txt", "data/val/5.txt", "data/val/7.txt"]
test_files = ["data/test/0.txt", "data/test/1.txt",
              "data/test/3.txt", "data/test/5.txt", "data/test/7.txt"]
dynamic_val_files = ["data/dynamic_val/0.txt", "data/dynamic_val/1.txt",
                     "data/dynamic_val/3.txt", "data/dynamic_val/5.txt", "data/dynamic_val/7.txt"]
dynamic_test_files = ["data/dynamic_test/0.txt", "data/dynamic_test/1.txt",
                      "data/dynamic_test/3.txt", "data/dynamic_test/5.txt", "data/dynamic_test/7.txt"]

all_files = train_files + val_files + test_files + \
    dynamic_val_files + dynamic_test_files
# print(all_files)


def get_article_path():
    path2docid2sentid = {}
    for filename in all_files:
        for line in open(filename, "r"):
            dline = ast.literal_eval(line)
            docid = dline["doc_id"]
            sentid = dline["sentence_id"]
            source = (docid.split("_")[0]).lower()
            source_dir = gigawords_dirs[source2dir[source]]
            source_filename = (docid.split(".")[0]).lower()[:-2]
            source_filename_path = f"{gigaword_path}/{source_dir}/data/{source}_eng/{source_filename}"
            # print(source_filename_path)
            if source_filename_path not in path2docid2sentid:
                path2docid2sentid[source_filename_path] = {}
            if docid not in path2docid2sentid[source_filename_path]:
                path2docid2sentid[source_filename_path][docid] = []
            path2docid2sentid[source_filename_path][docid].append(sentid)
    return path2docid2sentid


def process_articles(path2docid2sentid):
    sentid2tokens = {}
    for path in tqdm(path2docid2sentid):
        docids = set(path2docid2sentid[path].keys())
        content = open(path).read()
        soup = BeautifulSoup(content, 'lxml')
        for doc in soup.find_all('doc'):
            docid = doc.attrs["id"]
            if docid not in docids:
                continue
            paras = doc.find_all('p')
            sentid = 0
            for p in paras:
                tmp = p.text
                tmp = " ".join(tmp.replace("\n", " ").strip().split())
                sentences = nlp(tmp).sents

                for sentence in sentences:
                    sentence = str(sentence).strip()
                    #print(sentid, sentence)
                    if sentence == "":
                        continue
                    sentid += 1
                    if sentid not in path2docid2sentid[path][docid]:
                        continue
                    sentence = nlp(sentence)

                    tokens = (str(sentence)).split(" ")
                    sentid2tokens[(docid, sentid)] = tokens

    return sentid2tokens


def output_with_text(input_file, prefix, sentid2tokens):
    fout = open(f"{output_dir}/{prefix}{input_file.split('/')[-1]}", "w")
    inputlines = open(input_file).read().strip().split("\n")
    for line in inputlines:
        sentence = ast.literal_eval(line)
        docid = sentence["doc_id"]
        sentid = sentence["sentence_id"]
        # if (docid, sentid) not in sentid2tokens:
        #    continue
        tokens = sentid2tokens[(docid, sentid)]
        sentence["tokens"] = tokens
        head_start, head_end = sentence["h"]["pos"][0], sentence["h"]["pos"][1]
        tail_start, tail_end = sentence["t"]["pos"][0], sentence["t"]["pos"][1]
        sentence["h"]["name"] = tokens[head_start:head_end]
        sentence["t"]["name"] = tokens[tail_start:tail_end]
        json.dump(sentence, fout)
        fout.write(",\n")
    fout.close()


print("collect all data paths")
sys.stdout.flush()
path2docid2sentid = get_article_path()
print(path2docid2sentid)
print("tokenizing")
sys.stdout.flush()
sentid2tokens = process_articles(path2docid2sentid)
print(sentid2tokens)
print("Tokenized")
sys.stdout.flush()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filepath in train_files:
    output_with_text(filepath, "", sentid2tokens)
for filepath in val_files:
    output_with_text(filepath, "val_", sentid2tokens)
for filepath in test_files:
    output_with_text(filepath, "test_", sentid2tokens)
for filepath in dynamic_val_files:
    output_with_text(filepath, "dynamic_val_", sentid2tokens)
for filepath in dynamic_test_files:
    output_with_text(filepath, "dynamic_test_", sentid2tokens)

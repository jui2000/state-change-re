import sys
import spacy
import json, ast
from os import listdir
from bs4 import BeautifulSoup
from tqdm import tqdm

input_path = sys.argv[1]#"/mnt/nfs/scratch1/dongxuzhang/dataset/LDC2011T07/"
output_path = sys.argv[2]

spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')

dirs = ["gigaword_eng_5_d1", "gigaword_eng_5_d2", "gigaword_eng_5_d3"]
newspapers = {"afp":0, "apw":0, "cna":1, "ltw":1, "nyt":1, "wpb":1, "xin":2}

output_folder = "/mnt/nfs/scratch1/jbshah/final_data/"
datasets = ["-11_10_","-31_30_","-101_100_", "-301_300_"]
relations = ["0","1","3","4","5","6","7"]

print("START")
for data in datasets: #This part is reading the final output lines
    for relation in relations:
        masked_file = open(output_folder + data + relation + "_train.txt","r")
        for masked_line in masked_file:
            masked_mod_line = ast.literal_eval(masked_line)
            masked_docid = masked_mod_line['doc_id']
            masked_sentenceid = masked_mod_line['sentence_id'] #By here we finally have the 2 IDS we need to match

            #open the doc_id that needs to be read based on masked_docid
            open_newpaper = (masked_docid.split("_")[0]).lower()

            with open(output_path, 'a') as json_file: #Just open the output file
                
                fil_path = input_path + dirs[newspapers[open_newpaper]] + "/data/" + open_newpaper + "_eng/" + (masked_docid.split(".")[0]).lower()

                content = open(fil_path[:-2]).read()
                soup = BeautifulSoup(content, 'lxml')
                docs = soup.find_all('doc')

                for doc in (docs):
                    doc_id = doc.attrs["id"]
                    if masked_docid !=doc_id:
                        continue
                    
                    sentence_id = 0
                    text = doc.find('text').text
                    paras = doc.find_all('p')                    
                    for p in paras:
                        tmp = p.text
                        tmp = " ".join(tmp.replace("\n"," ").strip().split())
                        sentences = nlp(tmp).sents
                        
                        for sentence in sentences:
                            en = []
                            sentence = str(sentence).strip()
                            if sentence == "":
                                continue
                            sentence_id += 1
                            if masked_sentenceid!=sentence_id:
                                continue
                            
                            sentence = nlp(sentence)

                            for ent in sentence.ents:
                                en.append([ent.text, ent.start_char, ent.end_char, ent.label_])
                            
                            if len(en)==0:
                                continue
                            
                            splited_text = (str(sentence)).split(" ")

                            masked_mod_line["tokens"] = splited_text
                            masked_mod_line["h"]["name"] = splited_text[masked_mod_line["h"]["pos"][0]:masked_mod_line["h"]["pos"][1]]

                            masked_mod_line["t"]["name"] = splited_text[masked_mod_line["t"]["pos"][0]:masked_mod_line["t"]["pos"][1]]
                            
                            json.dump(masked_mod_line, json_file)
                            json_file.write(',\n')
        

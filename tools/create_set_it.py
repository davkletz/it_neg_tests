import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2" # usa una sola GPU, la num 1 (0,1); da mettere prim di import torch
from joblib import load, dump # x salvare e recuperare roba da e su cartelle, vd dump
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch # lib x mod ML, calcoli su vettor/tensori
from random import random, seed, shuffle


from build_array import build_array, build_hypo, build_masked_context
from mask_prediction import make_and_encode_batch

#from tools.chech_conjug import check_conjugation




#path = r"C:\Users\Viola\Desktop\Scambio_Poibeau\Esperimento_neg_eval\inputs"
path = r"../Inputs"
path = "Inputs"
'''fName_file_path = f"{path}\100_names_f.txt"
mName_file_path = f"{path}\100_names_m.txt"
fProf_file_path = f"{path}\100_mestieri_f.txt"
mProf_file_path = f"{path}\100_mestieri_m.txt"
hypo_file_path = f"{path}\frasi_it.txt"
'''
fName_file_path = f"{path}/100_names_f.txt"
mName_file_path = f"{path}/100_names_m.txt"
fProf_file_path = f"{path}/100_mestieri_f.txt"
mProf_file_path = f"{path}/100_mestieri_m.txt"
hypo_file_path = f"{path}/frasi_it.txt"


size_batches = 100 # dimensione batch x ciclo di calcolo

def main():

    #for model_name in [ "flaubert/flaubert_base_cased", "camembert-base", "bert-base-multilingual-cased",
    #                   "camembert/camembert-large", "flaubert/flaubert_large_cased"]:

    # f = open(file.txt)
    # a = f.readlines()
    # listverb = []
    # for verb in a:
    #   listverb.append(verb[:-1])
    # dump(listverb, "path\*.joblib")
    # load x richiamare

    #for model_name in ["dbmdz/bert-base-italian-cased", "bert-base-multilingual-cased",
    #                   "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", "Musixmatch/umberto-commoncrawl-cased-v1"]:
    for model_name in ["dbmdz/bert-base-italian-xxl-cased"]:

        seed(42)

        if model_name in ["dbmdz/bert-base-italian-cased", "dbmdz/bert-base-italian-xxl-cased"]:
            list_verbs = load(f"{path}\base_verbs.joblib")

        elif model_name in ["bert-base-multilingual-cased"]: # joblib
            list_verbs = load(f"{path}\multilg_verbs.joblib")

        elif model_name in ["m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"]: #joblib
            list_verbs = load(f"{path}\alberto_verbs.joblib")
            #list_verbs = load(f"{path}/fake_monovalent_monotokenized_flaubert.joblib")

        elif model_name in ["Musixmatch/umberto-commoncrawl-cased-v1"]: #joblib
            list_verbs = load(f"{path}/umberto_verbs.joblib")

        elif model_name in ["dbmdz/bert-base-italian-xxl-cased"]:
            list_verbs = load(f"{path}/base_verbs.joblib")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # cuda = GPU, cerca una GPU su cui mandare, altrim su cpu
        model.to(device) # manda su gpu, occhio che sia tutto in un posto solo
        for param in model.parameters():
            param.requires_grad = False # non salvare il gradiente a ogni step della backprop altrim tiene troppo spazio
        
        # modo normale per farlo:
        # with torch.no_grad():
            # tutte le cose del modello sotto questo comando e non salva niente

        # controllare come si indica il carattere mask
        mask_token = tokenizer.mask_token
        print(f"Mask token: {mask_token}") # ti dà direttamente per il modello come quel modello indica il mask token

        fName_file = open(fName_file_path, "r")
        mName_file = open(mName_file_path, "r")
        fProf_file = open(fProf_file_path, "r")
        mProf_file = open(mProf_file_path, "r")


        professionsarray = {"f": build_array(fProf_file), "m": build_array(mProf_file)} # buildarray è def in un altro file .py, crea lista da file txt
        fnamearray = build_array(fName_file)
        mnamearray = build_array(mName_file)


        name_arrays = {"f": fnamearray, "m": mnamearray}
        pronouns_maj = {"f": "Lei", "m": "Lui"}

        list_good_patterns_model = [] # quelli che nelle fr afferm ripetono il vb nel mask, triplette nome-mest-vb

        total_sentences = 0
        tot_good_preds = 0
        detail_verbs = {v : 0 for v in list_verbs} # vedere ogni vb qunte volt si ripete


        for gender in ["f", "m"]:
            # gender adaptations
            current_pronouns_maj = pronouns_maj[gender]

            for name_available in name_arrays[gender]:  # each of the top 100 names

                batch_sentences = [] # batch = x non mandare una frase alla volte alla gpu, ne prendo un mucchietto (batch) e faccio i calcoli per un gruppo alla volta
                batch_verbs = []
                for profession_available in professionsarray[gender]:
                    current_list_verbs = list_verbs.copy()
                    shuffle(current_list_verbs)

                    found = False # quando ne trovo uno mi fermo

                    for verb_available in current_list_verbs:
                        #print(f"current verb : {verb_available}")
                        #if not complete_check and found:
                        #    break

                        
                        current_sentence = build_masked_context(name_available, profession_available, verb_available, current_pronouns_maj, mask_token)

                        #print(current_sentence)
                        #quit()

                        batch_sentences.append(current_sentence)
                        batch_verbs.append(verb_available)
                        total_sentences += 1

                        if total_sentences % 1000 == 0:
                            print(f"current : {total_sentences}, {len(list_good_patterns_model)}")

                        if len(batch_sentences) == size_batches:
                            new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_sentences, tokenizer, model, device, batch_verbs, name_available, profession_available, current_pronouns_maj, found) # registra le frasi che vanno
                            tot_good_preds+=nb_good_pred
                            if new_sentence!= None:
                                list_good_patterns_model.append(new_sentence)
                            batch_sentences = []
                            batch_verbs = []
                            for found_verb in found_verbs:
                                detail_verbs[found_verb] +=  1 # x ogni vb quante volte si ripete



                    if len(batch_sentences) > 0: # ripete nel caso il num di frasi non sia divis x 100
                        new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_sentences, tokenizer, model, device, batch_verbs, name_available, profession_available, current_pronouns_maj, found)

                        tot_good_preds += nb_good_pred
                        if new_sentence != None:
                            list_good_patterns_model.append(new_sentence)
                        batch_sentences = []
                        batch_verbs = []
                        for found_verb in found_verbs:
                            detail_verbs[found_verb] += 1





        print(f"\n\nname model : {model_name}")
        print(f"total good predictions : {tot_good_preds}")

        print(f"total sentences : {total_sentences}")
        print(f"detail verbs : {detail_verbs}")




        if "/" in model_name:
            current_model_name = model_name.split("/")[-1]
            # current_model_name = current_model_name[0] + "_" + current_model_name[1]

        else:
            current_model_name = model_name

        if not os.path.exists(f"{path}/{current_model_name}"):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs(f"{path}/{current_model_name}")

        # salva le cose trovate
        dump(list_good_patterns_model, f"{path}/{current_model_name}/list_good_patterns_mono_{current_model_name}.joblib")
        dump(detail_verbs, f"{path}/{current_model_name}/detail_verbs_mono_{current_model_name}.joblib")
        dump(tot_good_preds, f"{path}/{current_model_name}/tot_good_preds_mono_{current_model_name}.joblib")
        dump(total_sentences, f"{path}/{current_model_name}/total_sentences_mono_{current_model_name}.joblib")




if __name__ == '__main__':
    main()





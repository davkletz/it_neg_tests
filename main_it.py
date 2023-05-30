import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tools.build_array import build_array, build_hypo, build_masked_sentences, build_masked_context
#from tools.extract_top_token import extract_top_token
from tools.mask_prediction import mask_prediction
from random import random, seed
from TG_comm import launchFct
from tools.chech_conjug import check_conjugation, get_conj
#from tools.comparisons import get_distrib
import numpy as np
from joblib import load
import sys



#path_sentences = "/home/dkletz/tmp/pycharm_project_99/2022-23/neg-eval-set/evaluation_script/Inputs"
path_sentences = "/home/dkletz/tmp/pycharm_project_99/stages/viola/it_neg_tests/Inputs"
## verificare solo cosa sostituisce al mask se non metto il contesto
dico_base_n = {"dbmdz/bert-base-italian-cased" : {"Lui" : "dorme", "Lei" : "dorme"},
            "bert-base-multilingual-cased" : {"Lui" : "è", "Lei" : "è"},
            "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0" : {"Lui" : "anche", "Lei" : "anche"}}


dico_base_p = {"dbmdz/bert-base-italian-cased" : {"Lui" : "lavora", "Lei" : "lavora"},
            "bert-base-multilingual-cased" : {"Lui" : "è", "Lei" : "lei"},
            "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0" : {"Lui" : "anche", "Lei" : "anche"}}

#results_repet_n = {model : 0 for model in dico_base_n.keys()}
#results_repet_p = {model : 0 for model in dico_base_p.keys()}


def main():
    #for model_name in ["dbmdz/bert-base-italian-cased", "bert-base-multilingual-cased",
    #                   "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"]:
    #for model_name in ["dbmdz/bert-base-italian-xxl-cased"]:
    for model_name in ["Musixmatch/umberto-commoncrawl-cased-v1"]:

        seed(42)

        if "/" in model_name:
            current_model_name = model_name.split("/")[-1]
            # current_model_name = current_model_name[0] + "_" + current_model_name[1]

        else:
            current_model_name = model_name

        list_good_patterns_model = load( f"{path_sentences}/{current_model_name}/list_good_patterns_mono_{current_model_name}.joblib")


        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        for param in model.parameters():
            param.requires_grad = False

        mask_token = tokenizer.mask_token
        print(f"Mask token: {mask_token}")

        #list_combi = ["Cp_Tp", "Cp_Tm", "Cm_Tp", "Cm_Tm"]


        #hypo_sentence_cons = "NOM è MET che ha l'abitudine di ACTION. PRON_maj non MASK molto spesso."
        #hypo_sentence_cons = "NOM è MET che non ha l'abitudine di ACTION. PRON_maj MASK molto spesso."
        #hypo_sentence_cons = "NOM è MET che non ha l'abitudine di ACTION. PRON_maj non MASK molto spesso."
        hypo_sentence_cons = sys.argv[1]

        # frase doppia neg?
        #hypo_sentence_cons = "NOM è MET che ha l'abitudine di ACTION. PRON_maj MASK davvero molto spesso."
        #hypo_sentence_cons = "PRON_maj MASK molto spesso."

        #Cm_Tp = "NOM est MET qui n'a l'habitude de ACTION. PRON_maj MASK vraiment souvent."

        '''if "PRON_maj non MASK" in hypo_sentence_cons:
            dico_base = dico_base_n
            results_repet = results_repet_n

        else:
            dico_base = dico_base_p
            results_repet = results_repet_p
        '''



        pronouns = {"Lei": "lei", "Lui": "lui"}


        total_sentences = 0

        total_repetition = 0


        rg_act_tok = []
        proba_act_tok = []

        print('startMMMM')


        for pattern in list_good_patterns_model:

            #print(torch.cuda.memory_allocated(device))
            name_available = pattern["name_available"]
            profession_available = pattern["profession_available"]
            verb_available = pattern["verb"]
            current_pronouns_maj = pattern["current_pronouns_maj"]

            conj_verb = get_conj(verb_available)
            token_verb_conj = tokenizer.encode(conj_verb)[1]




            #print(current_pronouns_maj)
            current_pronoun = pronouns[current_pronouns_maj]
            '''
            if verb_available[0] in ["a", "e", "i", "o", "u", "y", "é", "è", "ê", "à", "â", "ù", "û", "ô", "î", "ï", "ü"]:
                print(verb_available)
                print('HAHA')
                quit()
            else:
                current_hypo_sentence = hypo_sentence_cons
            '''

            current_hypo_sentence = hypo_sentence_cons # senza la distinz x vb con vocale o consonante
            masked_sentence = build_masked_sentences(current_hypo_sentence, name_available, profession_available,verb_available, 
                                                     current_pronouns_maj, current_pronoun, mask_token)



            prediction_conj, mask_token_logits, indice_act, proba_act = mask_prediction(masked_sentence, tokenizer, model, device, token_verb_conj)

            rg_act_tok.append(indice_act)
            proba_act_tok.append(proba_act.item())


            sente_pred = masked_sentence.replace(mask_token, prediction_conj)

            '''print(dico_base.keys())

            if prediction_conj == dico_base[model_name][current_pronouns_maj]:
               results_repet[model_name] += 1
            '''

            '''print(f"\nsente_pred : {sente_pred}")
            print(dico_base[model_name][current_pronouns_maj])
            print(prediction_conj)

            break'''

            if random() <0.0:

                print(f"\nsente_pred : {sente_pred}")
                #print(dico_base[model_name][current_pronouns_maj])
                #print(prediction_conj)


            if check_conjugation(verb_available, prediction_conj):
                total_repetition += 1
            '''else:
                print(f"\nsente_pred : {sente_pred}")
            '''

            total_sentences += 1

        print("\n=====================================\n")
        print(model_name)
        print(total_repetition)
        print(total_sentences)
        print((round((total_repetition/total_sentences)/100))*100)
        print("REPET BASE:")
        #print(results_repet[model_name])
        print(f"rg token : {np.mean(rg_act_tok)}")
        print(f"proba token : {np.mean(proba_act_tok)}")







        continue



    print('LMLMLMLM')
    #print(results_repet)






if __name__ == '__main__':
    launchFct(main)






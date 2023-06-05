import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from joblib import load, dump
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from sklearn.neural_network import MLPClassifier

import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from random import shuffle, seed

# from sklearn.model_selection import train_test_split


from mlconjug3 import Conjugator


########################
### useful functions ###
########################



conjugator = Conjugator(language="it") # così usa l'italiano

def check_conjugation(verb, conjugation):
    """
    Verifie si une forme est bien une flexion d'un verbe.

    Args:
        verb (str): The verb to check.
        conjugation (str): Forme a verifier.

    Returns:
        bool: Vrai si la forme est bien une flexion du verbe.

    """
    try:
        verb_conjs = conjugator.conjugate(verb).iterate()

    except:
        for k in range(10000):
            l = 0

        print(f'except : {verb}, {conjugation}')

        return False


    #return (('Indicativo', 'Indicativo presente', 'egli/ella', conjugation) in verb_conjs)
    return (('Indicativo', 'Indicativo presente', '3s', conjugation) in verb_conjs)


def build_masked_context(name_available, profession_available, verb, current_pronouns_maj, mask_token):
    #if verb[0] in ['a', 'e', 'i', 'o', 'u', 'h', "é", "è", "ê", "à", "â", "ô", "î", "ï", "û", "ù", "ü", "y"]:
    #    context_available = "NOM est MET qui a l'habitude d'ACTION. PRON_maj MASK vraiment souvent."
    #else:
    context_available = "NOM è MET che ha l'abitudine di ACTION. PRON_maj MASK molto spesso."# it

    new_context_sentence_available = context_available.replace('NOM', name_available)
    new_context_sentence_available = new_context_sentence_available.replace('MET', profession_available)
    new_context_sentence_available = new_context_sentence_available.replace('ACTION', verb)
    new_context_sentence_available = new_context_sentence_available.replace('PRON_maj', current_pronouns_maj)
    new_context_sentence_available = new_context_sentence_available.replace('MASK', mask_token)




    return new_context_sentence_available

def build_array(sourcefile):
    """"
    The function builds an array out of an inputfile,
    one array entry per line
    """
    array_in_construction=[]
    for line in sourcefile:
        #print(line)
        cleanline=line.strip('\n ')
        array_in_construction.append(cleanline)
    return array_in_construction


def make_and_encode_batch(current_batch, tokenizer, model, device, batch_verbs, name_available, profession_available,
                          current_pronouns_maj, found):
    current_found = found  # true if the current batch contained good sentences
    good_pred = 0
    detail_verbs = []

    # get the predicted tokens for the batch of sentences

    predictions = encode_batch(current_batch, tokenizer, model, device)
    new_sentence = None

    # for each prediction, check if the model predicted the same verb that was in the context sentence
    for i, prediction_available in enumerate(predictions):
        good_verb = batch_verbs[i]  # the desired verb
        print('\n')
        print(good_verb)
        print(prediction_available)

        if check_conjugation(good_verb, prediction_available):
            # outputs True value if the prediction is the 3rd person plural of the desired verb
            detail_verbs.append(good_verb)
            good_pred += 1
            good_dico = {"name_available": name_available, "profession_available": profession_available,
                         "verb": good_verb, "current_pronouns_maj": current_pronouns_maj,
                         "masked_prediction": prediction_available}

            if not current_found:
                # once a good sentence is found, the "found" value is set to true
                # and the "new_sentence" value is the dictionary of all elements in the sentence
                new_sentence = good_dico

                current_found = True
                # if not complete_check: ########
                #    break
    return new_sentence, current_found, good_pred, detail_verbs


def encode_batch(current_batch, tokenizer, model, device):
    with torch.no_grad():
        # encode sentences
        encoded_sentence = tokenizer.batch_encode_plus(current_batch, padding=True, return_tensors="pt").to(device)
        # get the mask-token index in the sentence
        mask_tokens_index = torch.where(encoded_sentence['input_ids'] == tokenizer.mask_token_id)
        # print(mask_tokens_index)
        # get logits vectors
        tokens_logits = model(**encoded_sentence)
        # print(tokens_logits)
        # print(tokens_logits['logits'].shape)

        # get the mask-token logit
        mask_tokens_logits = tokens_logits['logits'][mask_tokens_index]
        # print(mask_tokens_logits.shape)

        # get the k highest logits
        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices  # .tolist()
        # print(top_tokens)

        # decode the batch of tokens, i.e. get the predicted tokens (each token is represented by an index in the vocabulary)
        predicted_tokens = tokenizer.batch_decode(top_tokens)
        # print(predicted_tokens)

    return predicted_tokens




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_mask = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-italian-cased')

model_mask = model_mask.to(device)


# select the italian model to test
model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased')
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')

###########################
### masked template set ###
###########################

# load names, professions and verbs for the templates
path = r"../Inputs"
fName_file_path = f"{path}/100_names_f.txt"
mName_file_path = f"{path}/100_names_m.txt"
fProf_file_path = f"{path}/100_mestieri_f.txt"
mProf_file_path = f"{path}/100_mestieri_m.txt"
hypo_file_path = f"{path}/frasi_it.txt"

fName_file = open(fName_file_path, "r")
mName_file = open(mName_file_path, "r")
fProf_file = open(fProf_file_path, "r")
mProf_file = open(mProf_file_path, "r")

list_verbs = load(f"{path}/base_verbs.joblib")

# dictionaries of names, professions and pronouns indexed by gender for template construction
professionsarray = {"f": build_array(fProf_file),
                    "m": build_array(mProf_file)}  # buildarray is a function for creating lists from txt files
fnamearray = build_array(fName_file)
mnamearray = build_array(mName_file)
name_arrays = {"f": fnamearray, "m": mnamearray}
pronouns_maj = {"f": "Lei", "m": "Lui"}

# set up list for patterns that, for the CpTp setting, predict for the mask the same verb that was in the context
list_good_patterns_model = []

total_sentences = 0  # counts tried sentences
tot_good_preds = 0  # counts sentences with repetition
detail_verbs = {v: 0 for v in
                list_verbs}  # counts, for each verb, how many times it is repeated in the mask if present in context

size_batches = 100

for gender in ["f", "m"]:
    current_pronouns_maj = pronouns_maj[gender]

    for name_available in name_arrays[gender]:
        batch_sentences = []  # batch of sentences to try in this cycle
        batch_verbs = []  # batch of verbs to try in this cycle

        for profession_available in professionsarray[gender]:

            current_list_verbs = list_verbs.copy()
            shuffle(current_list_verbs)

            found = False  # to stop when a good verb is found

            for verb_available in current_list_verbs:
                # print(f"current verb : {verb_available}")
                # if not complete_check and found:
                #    break

                current_sentence = build_masked_context(name_available, profession_available, verb_available,
                                                        current_pronouns_maj, mask_token=tokenizer.mask_token)

                # print(current_sentence)
                # quit()

                batch_sentences.append(current_sentence)
                batch_verbs.append(verb_available)
                total_sentences += 1

                if total_sentences % 1000 == 0:
                    print(f"current : {total_sentences}, {len(list_good_patterns_model)}")

                # get the result at the end of the batch
                if len(batch_sentences) == size_batches:
                    new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_sentences, tokenizer,
                                                                                           model_mask, device,
                                                                                           batch_verbs, name_available,
                                                                                           profession_available,
                                                                                           current_pronouns_maj, found)
                    tot_good_preds += nb_good_pred
                    if new_sentence != None:
                        list_good_patterns_model.append(new_sentence)
                    batch_sentences = []
                    batch_verbs = []
                    for found_verb in found_verbs:
                        detail_verbs[found_verb] += 1  # add one repetition to the count for the found verb

            # repetition for what is left out of the last batch
            if len(batch_sentences) > 0:
                new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_sentences, tokenizer,
                                                                                       model_mask, device, batch_verbs,
                                                                                       name_available,
                                                                                       profession_available,
                                                                                       current_pronouns_maj, found)

                tot_good_preds += nb_good_pred
                if new_sentence != None:
                    list_good_patterns_model.append(new_sentence)
                batch_sentences = []
                batch_verbs = []
                for found_verb in found_verbs:
                    detail_verbs[found_verb] += 1

# create the CpTp set
template_sentences_pos = []
for pattern in list_good_patterns_model:
    # build sentences putting the conjugated verb instead of the mask
    sent = build_masked_context(pattern["name_available"], pattern["profession_available"],
                                pattern["verb"], pattern["current_pronouns_maj"], pattern["masked_prediction"])
    template_sentences_pos.append(sent)

# create the CnTn set
template_sentences_neg = []
pat_and_repl = [[r"che ha", "che non ha"], [r" Lei ", " Lei non "], [r" Lui ", " Lui non "]]

for sent in template_sentences_pos:
    sent_neg = sent
    for pair in pat_and_repl:
        sent_neg = re.sub(pair[0], pair[1], sent_neg)
    template_sentences_neg.append(sent_neg)

# extract CLS for each template sentence
# for each set of sentences, we encode each sentence
for sent_list in [template_sentences_neg, template_sentences_pos]:
    batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt")

    # then extract only the outputs for each sentence
    with torch.no_grad():
        tokens_outputs = model(**batch_encoded)

    # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
    cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

    cls_encodings = cls_encodings.cpu().numpy()

    if sent_list == template_sentences_neg:
        cls_temp_neg = cls_encodings
    elif sent_list == template_sentences_pos:
        cls_temp_pos = cls_encodings

cls_temp_neg.shuffle()
cls_temp_pos.shuffle()

cls_temp_pos = cls_temp_pos[:size_test]
cls_temp_neg = cls_temp_neg[:size_test]




############################
### masked template test ###
############################


train_temp = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]))
train_temp_lab = np.concatenate((np.zeros(train_size), np.ones(train_size)))
test_temp = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]))
test_temp_lab = np.concatenate((np.zeros(test_size), np.ones(test_size)))

scaler.fit(train_temp)
train = scaler.transform(train_temp)
test_2 = scaler.transform(test_temp)










##############################
### paisa "non" extraction ###
##############################


size_test = 10000


# upload the Italian corpus
with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
    paisa = infile.read()

# from the corpus, select all texts containing "wiki" in their tag's url
wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n"
paisa_wiki = re.findall(wiki_pattern, paisa)
# print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")


# pattern for finding whole sentences in the texts (defined by the capital letter in the beginning, the period at the end and a minimum length)
sent = []
pattern = r" [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds kind of acceptable sentences

for text in paisa_wiki:
    found = re.findall(pattern, text)
    for elem in found:
        if len(elem) > 25:
            sent.append(elem)
    if len(sent) > size_test * 100:
        break

print(f"Number of sentences: {len(sent)}")


# splitting the sentences above into two lists:
sent_pos = []
sent_neg = []

# pattern to find the negation in a sentence
neg_patt = r"\b[Nn]on\b"

for s in sent:
    matches = re.search(neg_patt, s)
    if matches:
        sent_neg.append(s)
    else:
        sent_pos.append(s)

size_test = min(size_test, len(sent_neg), len(sent_pos))

shuffle(sent_neg)
shuffle(sent_pos)

# select a fixed numb of sentences to test
sent_neg = sent_neg[:size_test]
sent_pos = sent_pos[:size_test]

### extract CLS
# for each set of sentences, we encode each sentence
for sent_list in [sent_neg, sent_pos]:
    batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt")

    # then extract only the outputs for each sentence
    with torch.no_grad():
        tokens_outputs = model(**batch_encoded)

    # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
    cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

    cls_encodings = cls_encodings.cpu().numpy()

    if sent_list == sent_neg:
        cls_encodings_neg = cls_encodings
    elif sent_list == sent_pos:
        cls_encodings_pos = cls_encodings

# train = torch.zeros(cls_encodings_neg.shape[0]*2, cls_encodings_neg.shape[1])
# train[cls_encodings_neg.shape[0]] = cls_encodings_neg[:9000]
# train = train.append(cls_encodings_pos[:9000])

# we use 90% of data as training and 10% as test
train_size = round(size_test * 0.9)
train = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]), 0)
labels = np.concatenate((np.zeros(train_size), np.ones(train_size)))
test = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]), 0)
test_size = int(size_test - train_size)
test_lab = np.concatenate((np.zeros(test_size), np.ones(test_size)))

# data normalization
scaler = StandardScaler()
scaler.fit(train)
dati_scaled = scaler.transform(train)

X = dati_scaled
test = scaler.transform(test)
# print(test)
# print(test_lab)

y = labels



########################################
### classifier creation and training ###
########################################


paisa_result = []
template_result = []

# set up the MLP classifier
# solver : adam or sgd
# hidden_layer_sizes : 40,40 or 350,350
# alpha : between 1e-5 and 1e-2
for hl in [(40, 40), (350, 350)]:
    for a in [1e-2, 1e-3, 1e-4, 1e-5]:
        for solv in ["adam", "sgd"]:
            clf = MLPClassifier(solver="adam", alpha=a,
                                hidden_layer_sizes=hl, random_state=1)

            # train on data
            clf.fit(X, y)

            # see predictions on the dataset
            clf.predict(test)
            right_pred = clf.score(test, test_lab)
            paisa_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\n")

            clf.predict(test_2)
            right_pred = clf.score(test_2, test_temp_lab)
            template_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\n")

print("PAISA' TEST\n\n")
for scores in paisa_result:
    print(scores)

print("TEMPLATE TEST\n\n")
for scores in template_result:
    print(scores)

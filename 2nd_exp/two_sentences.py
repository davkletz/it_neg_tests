import re
from random import shuffle, seed
import nltk

seed(42)


with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
  paisa = infile.read()



wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url

paisa_wiki = re.findall(wiki_pattern, paisa)

print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")


i = 0
print(len(paisa_wiki))
shuffle(paisa_wiki)
paisa_wiki = paisa_wiki[:5000]


nb_sents_to_catch = 5000

sent = []



Cn_patt = r"\b[Nn]on\b.*\..*\."  # finds the negation in a sentence
Tn_patt = r"\..*\b[Nn]on\b.*\."  # finds the negation in a sentence


Cn = []
CpTp = []
CpTn = []


CnTp = []
CnTn = []


for text in paisa_wiki:

    if len(CnTn) >= nb_sents_to_catch and len(CnTp) >= nb_sents_to_catch and len(CpTn) >= nb_sents_to_catch and len(CpTp) >= nb_sents_to_catch:
        break


    if i%100 == 0:
        print(f"Text number {i}, {len(sent)} sentences found")
    i += 1
    tokens = nltk.sent_tokenize(text)

    for element in range(len(tokens)-1):
        new_sent = tokens[element] + " " + tokens[element+1]


        if len(new_sent) > 25 and not "\n" in new_sent:


            matches1 = re.search(Cn_patt, new_sent)
            if matches1:
                double_neg = re.search(Tn_patt, new_sent)
                if double_neg:
                    CnTn.append(new_sent)
                else:
                    CnTp.append(new_sent)
            else:
                matches2 = re.search(Tn_patt, new_sent)
                if matches2:
                    CpTn.append(new_sent)
                else:
                    CpTp.append(new_sent)

print(f"Number of sentences: {len(sent)}")



print(f"Number of sent_CnTp sentences : {len(CnTp)} \n\nNumber of CpTn sentences : {len(CpTn)}\n\nNumber of CnTp sentences : {len(CnTp)}\n\nNumber of CnTn sentences : {len(CnTn)}\n\n")





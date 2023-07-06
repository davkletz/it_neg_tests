import re
from random import shuffle
import nltk


with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
  paisa = infile.read()



wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url

paisa_wiki = re.findall(wiki_pattern, paisa)

print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")


i = 0
print(len(paisa_wiki))
shuffle(paisa_wiki)
paisa_wiki = paisa_wiki[:5000]

sent = []


for text in paisa_wiki:
    if i%100 == 0:
        print(f"Text number {i}, {len(sent)} sentences found")
    i += 1
    tokens = nltk.sent_tokenize(text)

    for element in range(len(tokens)-1):
        new_sent = tokens[element] + " " + tokens[element+1]

        if i<=5:
            print(new_sent)
        if len(new_sent) > 25:
            sent.append(new_sent)

print(f"Number of sentences: {len(sent)}")




Cn_patt = r"\b[Nn]on\b.*\..*\."  # finds the negation in a sentence
Tn_patt = r"\..*\b[Nn]on\b.*\."  # finds the negation in a sentence


Cn = []
CpTp = []
CpTn = []


for s in sent:
  matches1 = re.search(Cn_patt, s)
  if matches1:
    Cn.append(s)
  else:
    matches2 = re.search(Tn_patt, s)
    if matches2:
       CpTn.append(s)
    else:
        CpTp.append(s)


CnTp = []
CnTn = []
for elem in Cn:
    double_neg = re.search(Tn_patt, elem)
    if double_neg:
        CnTn.append(elem)
    else:
        CnTp.append(elem)


print(f"Number of sent_CnTp sentences : {len(CnTp)} \n\nNumber of CpTn sentences : {len(CpTn)}\n\nNumber of CnTp sentences : {len(CnTp)}\n\nNumber of CnTn sentences : {len(CnTn)}\n\n")


shuffle(CnTp)

for k in range(10):
    print("\n")
    print(CnTp[k])

shuffle(CpTn)

for k in range(10):
    print("\n")
    print(CpTn[k])

shuffle(CnTn)

for k in range(10):
    print("\n")
    print(CnTn[k])

shuffle(CpTp)

for k in range(10):
    print("\n")
    print(CpTp[k])




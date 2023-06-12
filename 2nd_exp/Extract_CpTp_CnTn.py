with open(r"paisa.raw.utf8", encoding='utf8') as infile:
    
    paisa = infile.read()

import re

wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url

paisa_wiki = re.findall(wiki_pattern, paisa)

print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")

sent = []

pattern = r"(?<= )[A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b[A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds series of two sentences



for text in paisa_wiki:
  found = re.findall(pattern, text)
  for elem in  found:
    if len(elem) > 25:
      sent.append(elem)



print(f"Number of sentences: {len(sent)}")

sent_CpTn = []
sent_CnTp = []

Cn_patt = r"\b[Nn]on\b.*\..*\."  # finds the negation in a sentence
Tn_patt = r"\..*\b[Nn]on\b.*\."  # finds the negation in a sentence



for s in sent:
  matches1 = re.search(Cn_patt, s)
  if matches1:
    sent_CnTp.append(s)
  else:
    matches2 = re.search(Tn_patt, s)
    if matches2:
       sent_CpTn.append(s)

for elem in sent_CnTp:
  double_neg = re.search(Tn_patt, elem)
  if double_neg:
    sent_CnTp.remove(elem)




print(f"Number of positive sentences : {len(frasi_pos)}\n\nNumber of negative sentences : {len(frasi_neg)}")
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from transformers import AutoTokenizer, AutoModel
import re

model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased')
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')



with open(r"paisa.raw.utf8", encoding='utf8') as infile:
    paisa = infile.read()

wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url
paisa_wiki = re.findall(wiki_pattern, paisa)
#print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")




sent = []
pattern = r" [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds kind of acceptable sentences

for text in paisa_wiki:
  found = re.findall(pattern, text)
  for elem in  found:
    if len(elem) > 25:
      sent.append(elem)

#print(f"Number of sentences: {len(sent)}")





sent_pos = []
sent_neg = []

neg_patt = r"\b[Nn]on\b"  # finds the negation in a sentence

for s in sent:
  matches = re.search(neg_patt, s)
  if matches:
    sent_neg.append(s)
  else:
    sent_pos.append(s)

#print(f"Number of positive sentences : {len(frasi_pos)}\n\nNumber of negative sentences : {len(frasi_neg)}")


for sent_list in [sent_neg, sent_pos]:
  batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt")

  with torch.no_grad():
    tokens_outputs = model(**batch_encoded)

  cls_encodings = tokens_outputs.last_hidden_states[:,0,:]
  cls_encodings = cls_encodings_pos.cpu().numpy()

  if sent_list == sent_neg:
    cls_encodings_neg = cls_encodings
  elif sent_list == sent_pos:
    cls_encodings_pos = cls_encodings 
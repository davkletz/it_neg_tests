# -*- coding: utf-8 -*-
"""classifier_neg_cls.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NiZjC1IHQGuH3GvkV88CE0jur9GnUrZj
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
from random import shuffle

size_test = 100

model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased') # automodel for masked LM perché automodel e basta crea solo i vettori, gli embedding, per la frase; per LM invece ricava anche le prob di ogni parola nel vocab, ossia fa il language model
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')

with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
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
  if len(sent)> size_test*10:
    break

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


size_test = min(size_test, len(sent_neg), len(sent_pos))

shuffle(sent_neg)
shuffle(sent_pos)

sent_neg = sent_neg[:size_test]
sent_pos = sent_pos[:size_test]


for sent_list in [sent_neg, sent_pos]:
  batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt")

  with torch.no_grad():
    tokens_outputs = model(**batch_encoded )

  cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

  cls_encodings = cls_encodings.cpu().numpy()

  if sent_list == sent_neg:
    cls_encodings_neg = cls_encodings
  elif sent_list == sent_pos:
    cls_encodings_pos = cls_encodings



#train = torch.zeros(cls_encodings_neg.shape[0]*2, cls_encodings_neg.shape[1])
#train[cls_encodings_neg.shape[0]] = cls_encodings_neg[:9000]
#train = train.append(cls_encodings_pos[:9000])


train_size = round(size_test*0.9)
test_size = size_test - train_size

print(len(cls_encodings_pos[:train_size]))
train = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]), 0)
test = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]), 0)
#test = cls_encodings_pos[9000:]
#test = test.append(cls_encodings_neg[9000:])
labels = np.concatenate((np.zeros(train_size), np.ones(train_size)))

#test_lab = np.empty(2000)
#test_lab = np.where(test_lab[:1000], 0, 1)
test_lab = np.concatenate((np.zeros(len(test)), np.ones(len(test))))


#scaler = StandardScaler()
#scaler.fit(dati.values)
#dati_scaled = scaler.transform(dati.values)

#scaler.fit(train)
#dati_scaled = scaler.transform(train)


#train = dati_scaled
#labels = df["class"].values


print(train)
print(labels)

print(len(train))
print(len(labels))
X = train
y = labels

# solver : adam o sgd
# hidden_layer_sizes : testare diverse
# alpha : tra 1e-5 e 1e-2
for hl in [(40,40), (350,350)]:
  for a in [1e-2, 1e-3, 1e-4, 1e-5]:
    for solv in ["adam", "sgd"]:
      clf = MLPClassifier(solver = "adam", alpha = a,
                    hidden_layer_sizes=hl, random_state = 1)

      clf.fit(X, y)

      clf.predict(test)
      right_pred = clf.score(test, test_lab)
      print(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\n")
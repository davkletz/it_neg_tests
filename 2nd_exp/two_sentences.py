import re
from random import shuffle
import nltk


with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
  paisa = infile.read()



wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url

paisa_wiki = re.findall(wiki_pattern, paisa)

print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")

sent = []

pattern = r"(?<= )[A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b[A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds series of two sentences


i = 0
print(len(paisa_wiki))
shuffle(paisa_wiki)
#paisa_wiki = paisa_wiki[:100]


for text in paisa_wiki:
    tokens = nltk.sent_tokenize(text)
    print(len(tokens))



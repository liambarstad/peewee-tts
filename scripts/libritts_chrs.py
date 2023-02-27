import pickle
import os

chars = {}
for root, dirs, files in os.walk('data/utterance_corpuses/LibriTTS/dev-clean'):
    for f in files:
        if f[-3:] == 'txt':
            st = open(os.path.join(root, f), 'r').read()
            for char in st:
                if char not in chars:
                    chars[char] = ord(char)
print(''.join(chars.keys()))

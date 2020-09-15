# -*- coding: utf-8 -*-
"""
Author: Adriana STAN
september 2020
part of RECOApy tool
https://gitlab.utcluj.ro/sadriana/recoapy
"""

import keras
import os, sys
import numpy as np
import string
from models_CNN.config_lang import *




def main(lang, input_file, output_file):
    exclude = set(string.punctuation + string.digits)        

    model_path = 'models_CNN/'+lang.lower()+'_clean_28042020.csv_cnn_s2s.keras'
    model = keras.models.load_model(model_path)
#    model.compile(optimizer='adam')

    input_token_index = config_lang[lang.lower()]['input_token_index']
    target_token_index = config_lang[lang.lower()]['target_token_index']
    num_encoder_tokens = config_lang[lang.lower()]['num_encoder_tokens']
    num_decoder_tokens = config_lang[lang.lower()]['num_decoder_tokens']
    max_encoder_seq_length = config_lang[lang.lower()]['max_encoder_seq_length']
    max_decoder_seq_length = config_lang[lang.lower()]['max_decoder_seq_length']
    

    input_texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:]

    for line in lines:
      for wd in line.split():
        if wd not in input_texts:
            if all([ch in input_token_index for ch in wd]):
              s = ''.join(ch for ch in wd.lower() if ch not in exclude and ch in input_token_index)
              input_texts.append(s.lower().strip())

    nb_examples = len(input_texts)

    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    test_encoder_input_data = np.zeros(
        (nb_examples, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    test_decoder_input_data = np.zeros(
        (nb_examples, max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, test_input_text in enumerate(input_texts):
        for t, char in enumerate(test_input_text):
            test_encoder_input_data[i, t, input_token_index[char]] = 1.
                

    in_encoder = test_encoder_input_data[:nb_examples]
    in_decoder = np.zeros(
        (nb_examples, max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    in_decoder[:, 0, target_token_index["~"]] = 1
    print('predicting...')
    predict = np.zeros(
        (nb_examples, max_decoder_seq_length),
        dtype='float32')

    for i in range(max_decoder_seq_length - 1):
        predict = model.predict([in_encoder, in_decoder])
        predict = predict.argmax(axis=-1)
        predict_ = predict[:, i].ravel().tolist()
        for j, x in enumerate(predict_):
            in_decoder[j, i + 1, x] = 1

    decoded_dict = {}
    for seq_index in range(nb_examples):
        output_seq = predict[seq_index, :].ravel().tolist()
        decoded = []
        for x in output_seq:
            if reverse_target_char_index[x] == "!":
                break
            else:
                decoded.append(reverse_target_char_index[x])
        
        decoded_sentence = " ".join(decoded)
        decoded = decoded_sentence.strip()
        decoded_dict[input_texts[seq_index]] = decoded

    with open(output_file, 'w') as fout:
        for i in range(len(lines)):
            fout.write("%s|" %lines[i].strip())
            for wd in lines[i].strip().lower().split():
                wd_strip = ''.join(ch for ch in wd.lower() if ch not in exclude)
                if wd_strip in decoded:
                    fout.write("[%s] " %decoded_dict[wd_strip])
                else:
                    fout.write("[UNK] ")
            fout.write("\n")
    print ('\n'+"*"*20)
    print ("DONE! Wrote %d lines to %s..." %(len(lines), output_file))
    print ("*"*20+'\n')

if __name__ == '__main__':
    if len(sys.argv)!=4:
        print ("Please use the following command line arguments\n python g2p_cnn.py <language> <in_prompt_file> <out_file>.\nNow exiting!!")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

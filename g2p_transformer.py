#!/usr/bin/env python
# coding: utf-8

from keras_transformer import get_model, decode
from tensorflow import keras
import os, sys
import numpy as np
import string
from models_transformer.config_lang_tsf import config_lang_tsf
exclude = set(string.punctuation + string.digits)


def main(lang, input_file, output_file):
    exclude = set(string.punctuation + string.digits)        

    input_token_index = config_lang_tsf[lang.lower()]['input_token_index']
    target_token_index = config_lang_tsf[lang.lower()]['target_token_index']
    max_encoder_seq_length = config_lang_tsf[lang.lower()]['max_encoder_seq_length']
    params = config_lang_tsf[lang.lower()]['params']
    target_max_len = 50
    token_num = max(len(target_token_index), len(input_token_index))

    model = get_model(
        token_num= token_num,
        embed_dim=params['embed_dim'],
        encoder_num=params['encoder_num'],
        decoder_num=params['decoder_num'],
        head_num=params['head_num'],
        hidden_dim=params['hidden_dim'],
        dropout_rate=params['dropout_rate'],
        use_same_embed=False,
        embed_weights=np.random.random((token_num, params['embed_dim']))
    )    

    model_path = 'models_transformer/'+lang.lower()+'_clean_28042020.csv_transformer.keras'
    model.load_weights(model_path)

    input_texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:]

    for line in lines:
        for wd in line.strip().split():
            if wd not in input_texts:
                if all([ch in input_token_index for ch in wd.lower() if ch not in exclude]):
                    s = ''.join(ch for ch in wd.lower() if ch not in exclude)
                    if len(s):
                        input_texts.append([x for x in s.lower().strip()])
         

    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    test_encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in input_texts]
    test_encode_tokens = [tokens + ['<PAD>'] * (50 - len(tokens)) for tokens in test_encode_tokens]
    test_input = [list(map(lambda x: input_token_index[x], tokens)) for tokens in test_encode_tokens]

    print ("predicting ...")
    decoded = {}
    for i in range(len(test_input)):
        int_decoded =[]
        prediction = decode(
            model,
            test_input[i],
            start_token=target_token_index['<START>'],
            end_token=target_token_index['<END>'],
            pad_token=target_token_index['<PAD>'],
            max_len = token_num+2+5
        )

        wd = ''.join(input_texts[i])
        for j in range(1, len(prediction)):
            if reverse_target_char_index[prediction[j]] in [ '<PAD>' , '<END>', '<START>']:
                  break
            else:
                int_decoded.append(prediction[j])
        decoded[wd] = ' '.join(map(lambda x: reverse_target_char_index[x], int_decoded))

    print (decoded)
    with open(output_file, 'w') as fout:
        for i in range(len(lines)):
            fout.write("%s|" %lines[i].strip())
            for wd in lines[i].strip().lower().split():
                wd_strip = ''.join(ch for ch in wd.lower() if ch not in exclude)
                if wd_strip in decoded:
                    fout.write("[%s] " %decoded[wd_strip])
                else:
                    fout.write("[UNK] ")
            fout.write("\n")
         

    print ('\n'+"*"*20)
    print ("DONE! Wrote %d lines to %s..." %(len(lines), output_file))
    print ("*"*20+'\n')


if __name__ == '__main__':
    if len(sys.argv)!=4:
        print ("Please use the following command line arguments\n python g2p_transformer.py <language> <in_prompt_file> <out_file>.\nNow exiting!!")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

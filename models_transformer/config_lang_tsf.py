config_lang_tsf = {
    'en':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, '-': 3, 'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17, 'o': 18, 'p': 19, 'q': 20, 'r': 21, 's': 22, 't': 23, 'u': 24, 'v': 25, 'w': 26, 'x': 27, 'y': 28, 'z': 29},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'd': 5, 'd͡ʒ': 6, 'e': 7, 'f': 8, 'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'z': 23, 'æ': 24, 'ŋ': 25, 'ɑ': 26, 'ɒ': 27, 'ɔ': 28, 'ə': 29, 'ɚ': 30, 'ɛ': 31, 'ɜ': 32, 'ɝ': 33, 'ɡ': 34, 'ɪ': 35, 'ɹ': 36, 'ʃ': 37, 'ʊ': 38, 'ʌ': 39, 'ʒ': 40, 'θ': 41},
        'params': {'batch_size': 64, 'decoder_num': 4, 'dropout_rate': 0.01, 'embed_dim': 64, 'encoder_num': 3, 'head_num': 2, 'hidden_dim': 512, 'optimizer': 'adam'},
        'max_encoder_seq_length' : 45
    },
    'ro':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, '-': 3, 'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17, 'o': 18, 'p': 19, 'q': 20, 'r': 21, 's': 22, 't': 23, 'u': 24, 'v': 25, 'w': 26, 'x': 27, 'y': 28, 'z': 29, 'â': 30, 'î': 31, 'ă': 32, 'ș': 33, 'ț': 34},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'd': 5, 'e': 6, 'e̯': 7, 'f': 8, 'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'o̯': 17, 'p': 18, 'r': 19, 's': 20, 't': 21, 't͡s': 22, 't͡ʃ': 23, 'u': 24, 'v': 25, 'w': 26, 'z': 27, 'ə': 28, 'ɡ': 29, 'ɨ': 30, 'ʃ': 31, 'ʒ': 32, 'ʲ': 33},
         'params' : {'batch_size': 256, 'decoder_num': 2, 'dropout_rate': 0.05, 'embed_dim': 64, 'encoder_num': 3, 'head_num': 2, 'hidden_dim': 64, 'optimizer': 'adam'},
         'max_encoder_seq_length': 44
    },

    'de':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, "'": 3, '-': 4, 'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'n': 18, 'o': 19, 'p': 20, 'q': 21, 'r': 22, 's': 23, 't': 24, 'u': 25, 'v': 26, 'w': 27, 'x': 28, 'y': 29, 'z': 30, 'ß': 31, 'ä': 32, 'ö': 33, 'ü': 34},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'ã': 4, 'b': 5, 'ç': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'i̯': 13, 'j': 14, 'k': 15, 'l': 16, 'l̩': 17, 'm': 18, 'm̩': 19, 'n': 20, 'n̩': 21, 'o': 22, 'p': 23, 'r': 24, 's': 25, 't': 26, 'u': 27, 'u̯': 28, 'v': 29, 'y': 30, 'z': 31, 'ø': 32, 'ŋ': 33, 'ŋ̍': 34, 'ŋ̩': 35, 'œ': 36, 'ɐ': 37, 'ɐ̯': 38, 'ɔ': 39, 'ə': 40, 'ɛ': 41, 'ɡ': 42, 'ɪ': 43, 'ɪ̯': 44, 'ʀ': 45, 'ʁ': 46, 'ʃ': 47, 'ʊ': 48, 'ʊ̯': 49, 'ʏ': 50, 'ʒ': 51, 'ʔ': 52, 'χ': 53},
        'params' : {'batch_size': 64, 'decoder_num': 2, 'dropout_rate': 0.05, 'embed_dim': 64, 'encoder_num': 4, 'head_num': 2, 'hidden_dim': 32, 'optimizer': 'adam'},
        'max_encoder_seq_length': 63
    },
    'fr':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, "'": 3, 'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17, 'o': 18, 'p': 19, 'q': 20, 'r': 21, 's': 22, 't': 23, 'u': 24, 'v': 25, 'w': 26, 'x': 27, 'y': 28, 'z': 29, 'à': 30, 'á': 31, 'â': 32, 'è': 33, 'é': 34, 'ê': 35, 'ë': 36, 'î': 37, 'ï': 38, 'ô': 39, 'ù': 40, 'û': 41, 'ü': 42, 'ÿ': 43},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'd': 5, 'e': 6, 'f': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 's': 16, 't': 17, 'u': 18, 'v': 19, 'w': 20, 'y': 21, 'z': 22, 'ø': 23, 'œ': 24, 'ɑ': 25, 'ɑ̃': 26, 'ɔ': 27, 'ɔ̃': 28, 'ə': 29, 'ɛ': 30, 'ɛ̃': 31, 'ɡ': 32, 'ɥ': 33, 'ɲ': 34, 'ʁ': 35, 'ʃ': 36, 'ʒ': 37},
        'params' : {'batch_size': 64, 'decoder_num': 3, 'dropout_rate': 0.05, 'embed_dim': 64, 'encoder_num': 2, 'head_num': 2, 'hidden_dim': 128, 'optimizer': 'adam'},
        'max_encoder_seq_length' : 52
    },
    'pl':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, '-': 3, ':': 4, 'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'n': 18, 'o': 19, 'p': 20, 'q': 21, 'r': 22, 's': 23, 't': 24, 'u': 25, 'v': 26, 'w': 27, 'x': 28, 'y': 29, 'z': 30, 'ó': 31, 'ą': 32, 'ć': 33, 'ę': 34, 'ł': 35, 'ń': 36, 'ś': 37, 'ź': 38, 'ż': 39, 'ˈ': 40},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'ã': 4, 'b': 5, 'c': 6, 'd': 7, 'd͡z': 8, 'd͡ʐ': 9, 'd͡ʑ': 10, 'f': 11, 'i': 12, 'j': 13, 'j̃': 14, 'k': 15, 'l': 16, 'm': 17, 'm̥': 18, 'n': 19, 'p': 20, 'r': 21, 'r̥': 22, 's': 23, 't': 24, 'ṭ': 25, 't͡s': 26, 't͡ɕ': 27, 't͡ʂ': 28, 'u': 29, 'ũ': 30, 'v': 31, 'w': 32, 'w̃': 33, 'x': 34, 'z': 35, 'æ': 36, 'ŋ': 37, 'ɔ': 38, 'ɔ̃': 39, 'ɕ': 40, 'ɛ': 41, 'ɛ̃': 42, 'ɡ': 43, 'ɣ': 44, 'ɨ': 45, 'ɲ': 46, 'ʂ': 47, 'ʐ': 48, 'ʑ': 49, 'ʲ': 50},
        'params' : {'batch_size': 128, 'decoder_num': 2, 'dropout_rate': 0.05, 'embed_dim': 128, 'encoder_num': 3, 'head_num': 4, 'hidden_dim': 1024, 'optimizer': 'adam'},
        'max_encoder_seq_length': 32
    },
    'es':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, "'": 3, '-': 4, 'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'n': 18, 'o': 19, 'p': 20, 'q': 21, 'r': 22, 's': 23, 't': 24, 'u': 25, 'v': 26, 'w': 27, 'x': 28, 'y': 29, 'z': 30, 'á': 31, 'ã': 32, 'ç': 33, 'é': 34, 'í': 35, 'ñ': 36, 'ó': 37, 'õ': 38, 'ú': 39, 'û': 40, 'ü': 41},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'd': 5, 'e': 6, 'f': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'n̪': 14, 'o': 15, 'p': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'w': 21, 'x': 22, 'ð': 23, 'ŋ': 24, 'ɡ': 25, 'ɣ': 26, 'ɲ': 27, 'ɾ': 28, 'ʃ': 29, 'ʎ': 30, 'ʝ': 31, 'β': 32, 'θ': 33},
        'params' :  {'batch_size': 32, 'decoder_num': 4, 'dropout_rate': 0.05, 'embed_dim': 32, 'encoder_num': 2, 'head_num': 4, 'hidden_dim': 32, 'optimizer': 'adam'},
        'max_encoder_seq_length': 28
    },
    'cz':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, "'": 3, '-': 4, 'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'n': 18, 'o': 19, 'p': 20, 'q': 21, 'r': 22, 's': 23, 't': 24, 'u': 25, 'v': 26, 'w': 27, 'x': 28, 'y': 29, 'z': 30, 'á': 31, 'é': 32, 'í': 33, 'ó': 34, 'ú': 35, 'ý': 36, 'č': 37, 'ď': 38, 'ě': 39, 'ň': 40, 'ř': 41, 'š': 42, 'ť': 43, 'ů': 44, 'ž': 45},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'l̩': 13, 'm': 14, 'm̩': 15, 'n': 16, 'o': 17, 'p': 18, 'r': 19, 'r̝': 20, 'r̝̊': 21, 'r̩': 22, 's': 23, 't': 24, 't͡s': 25, 't͡ʃ': 26, 'u': 27, 'v': 28, 'x': 29, 'z': 30, 'ŋ': 31, 'ɔ': 32, 'ɛ': 33, 'ɟ': 34, 'ɡ': 35, 'ɦ': 36, 'ɪ': 37, 'ɲ': 38, 'ʃ': 39, 'ʊ': 40, 'ʊ̯': 41, 'ʒ': 42, 'ʔ': 43},
        'params':  {'batch_size': 32, 'decoder_num': 2, 'dropout_rate': 0.05, 'embed_dim': 32, 'encoder_num': 2, 'head_num': 2, 'hidden_dim': 64, 'optimizer': 'adam'},
        'max_encoder_seq_length': 33
     },
      'it':{
        'input_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, "'": 3, '-': 4, 'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'n': 18, 'o': 19, 'p': 20, 'q': 21, 'r': 22, 's': 23, 't': 24, 'u': 25, 'v': 26, 'w': 27, 'x': 28, 'y': 29, 'z': 30, 'à': 31, 'è': 32, 'é': 33, 'ì': 34, 'ò': 35, 'ù': 36, 'ú': 37},
        'target_token_index': {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'd': 5, 'e': 6, 'f': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'z': 22, 'ŋ': 23, 'ɔ': 24, 'ɛ': 25, 'ɡ': 26, 'ɲ': 27, 'ʃ': 28, 'ʎ': 29, 'ʒ': 30},
        'params':  {'batch_size': 64, 'decoder_num': 2, 'dropout_rate': 0.01, 'embed_dim': 64, 'encoder_num': 2, 'head_num': 2, 'hidden_dim': 512, 'optimizer': 'adam'},
        'max_encoder_seq_length': 25
      }
}
    


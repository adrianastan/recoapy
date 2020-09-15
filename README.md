# RECOApy

## Tool description

RECOApy streamlines the steps of data recording and pre-processing 
required in end-to-end speech-based applications. The tool implements an easy-to-use 
interface for prompted speech recording, spectrogram and waveform analysis, 
utterance-level normalisation and silence trimming, as well grapheme-to-phoneme 
conversion of the prompts in eight languages:  Czech, English, French, German, 
Italian, Polish, Romanian and Spanish.

The tool's description was accepted for publication at Interspeech 2020. If you
use the tool, please cite:
> Adriana STAN, *RECOApy: Data recording, pre-processing and phonetic transcription 
for end-to-end speech-based applications*, Proceedings of Interspeech 2020, Shanghai, China. \[[paper](https://arxiv.org/abs/2009.05493)\]


## Cleaned Wiktionary lexicons

The cleaned Wiktionary lexicons are available in: 

    wiktionary_lexicons/

## Using the G2P module

To use the G2P module, run:

`python g2p_cnn.py <lang> <input_file> <output_file>`  for the CNN-based models, or

`python g2p_transformer.py <lang> <input_file> <output_file>` for the Transformer-based models.

e.g.

`python g2p_cnn.py RO prompts/ivan.txt prompts/ivan_phonetic.txt`

The available language identifiers are:
- EN - English
- RO - Romanian
- FR - French
- DE - German
- IT - Italian
- CZ - Czech
- ES - Spanish
- PL - Polish

The `g2p_cnn.py` script takes an input file with one utterance per line, strips the non-alphabetic symbols, runs the CNN-based models and outputs a file in the format:

`Orthographic transcript | [p h o n e t i c] [t r a n s c r i p t]`

The models will output an  `[UNK]` token for the words which do not contain valid graphemes in the corresponding language. For a list of valid graphemes check the `models_cnn/config_lang.py` or `models_transformer/config_lang_tsf.py` files.

This file can then be used as input to the RECOApy tool.


## Running the RECOApy tool


1. Edit the `hyperparameters.py` file:
- `counter` refers to the file id from which the output wav files will be indexed
- `filename_id` is the name used in the output files
- `output_folder` where to store the recordings
- `max_seconds` - the maximum length of the recordings
- `plot_specs` wheter to display the spectrogram after each prompt recording
- `normalize_wavs` and `trim_wavs` mark the post-processing of the recordings for waveform normalisation and silence trimming.

2. Start the RECOApy tool:

    `python RECOApy.py`

3. Load the prompt file using the top menu. The prompt file may or may not contain the phonetic transcription. 
4. Do a short check of the input volume.
5. Start recording one prompt at a time and monitor the waveform and spectrogram to make sure that the volume is ok and that no clipping occurs.
6. If a recording may be correct but you are unsure, you can save a copy of it using the `Safe copy` option.
7. The output recordings will be in the `output_folder`



***

For any questions,
Adriana.Stan@com.utcluj.ro

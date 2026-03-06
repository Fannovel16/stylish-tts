# Stylish TTS

<a target="_blank" href="https://colab.research.google.com/github/Stylish-TTS/stylish-tts/blob/main/notebook/train-stylish-on-LJSpeech-Sidon.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/Stylish-TTS/stylish-tts/blob/main/notebook/reference-train-stylish.ipynb">
  <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/>
</a>

Stylish TTS is a lightweight text-to-speech system suitable for offline local use. Our goal is providing consistency for long form text and screen reading with a focus on high quality single speaker models rather than zero-shot voice cloning. The architecture was based on [StyleTTS 2](https://github.com/yl4579/StyleTTS2), but has now diverged substantially.

TODO: Make some samples

# Roadmap to Release

Stylish TTS is getting close to release. But before release, we need to complete these tasks:

- [ ] Verify final model architecture
- [x] Merge disc-opt into main
- [ ] Verify ONNX conversion
- [x] Rework CLI
- [x] Import pitch cache script and make it use a concurrent.futures worker pool
- [ ] Make sure it can work as a PyPi package
- [x] Do proper stage detection in dataloader to prevent mixups with precached alignment/pitch
- [ ] Replace checkpointing with safetensors instead of accelerator checkpoint
- [ ] Remove dependency on accelerator
- [ ] Audit dependencies
- [ ] Audit and fix any remaining torch warnings
- [ ] Move test_onnx to stylish-tts module and remake it into at least a barebones inferencer.

# Getting Started

## Dependencies

The biggest dependency is of course the nvidia/CUDA drivers/system.

After those are installed, you can run the training code using uv which will take care of all the other python dependencies. Install uv via:

```
pipx install uv
pipx ensure-path
```

# Training a Model

In order to train your model, you need a GPU (or CPU and plenty of time) with PyTorch support and you will need a dataset. You will need your own `config.yml` file created from a template [here](https://github.com/Stylish-TTS/stylish-tts/blob/main/config/config.yml). You will need to specify the device ("cuda" or "cpu" or whatever will work with your torch installation).

You will need to fill in the `dataset` section. The `path` should be the root of your dataset, and the various other paths in this section are relative to that root. If you use the default file and directory names, your directory structure will look something like this:

```
path/to/your/dataset/             # Root
|
+-> train-list.txt                # Training list file (described below)
|
+-> val-list.txt                  # Validation list file (described below)
|
+-> wav-dir                       # Folder with wav files for segments
|   |
|   +-> something.wav
|   |
|   +-> other.wav
|   |
|   +-> ...
|
+-> pitch.safetensors             # Pre-cached segment pitches. You will generate this file below
|
+-> alignment.safetensors         # Pre-cached alignments. You will generate this file below
|
+-> alignment_model.safetensors   # Model for generating alignments. You will train this below

```

## Preparing a dataset

A dataset consists of many segments. Each segment has a written text and an audio file where that text is spoken by a reader. 

### Segment Distribution

Segments must have 510 phonemes or less. Audio segments must be at least 0.25 seconds long. The upper limit on audio length is determined by your VRAM and the training stage. Generally speaking, you will want to have a distribution of segments between 0.25 seconds and 10 seconds long. If your range doesn't cover the shortest lengths, your model will sound worse when doing short utterances of one word or a few words. If your range doesn't cover longer lengths which include multiple sentences, your model will tend to skip past punctuation too quickly. If you have a the VRAM, you can include even longer segments, though there are diminishing returns.

### Training List / Validation List

Training and validation lists are a series of lines in the following format:

`<filename>|<phonemes>|<speaker-id>|<plaintext>`

The filename for the segment audio and should be a .wav file (24 khz, mono) in the wav_path from your config.yml.

The phonemes are the IPA representation of how your segment text is pronounced.

Speaker ID is an arbitrary integer which should be applied to every segment that has the same speaker. For single-speaker datasets, this will typically always be '0'.

The plaintext is the original text of your utterance before phonemization. It does not need to be tokenized or normalized, but obviously should not include the '|' character.

### Pitch Data

Stylish TTS uses a pre-cached ground truth pitch (F0) for all your segments. To generate these pitches, run:

```
cd stylish-tts
uv run stylish-train pitch /path/to/your/config.yml --workers 16
```

The number of workers should be approximately equal to the number of cores you have. By default, Harvest is used to extract pitch which is a CPU-based system. If you find this too slow, there is also a GPU-based option available by passing `--method rmvpe` on the command line. When finished, it will write the pre-cached pitches at the file specified by your `config.yml`.

### Alignment Data

Alignment data is also pre-cached and you will need to train an alignment model first to generate the pre-cached data. This is a multi-step process but only needs to be done once for your dataset after which you can just use the cached results similar to pitch data.

First, you run train.py using the special alignment stage. For a description of the other parameters, see below.

#### Expectations during alignment pre-training

In this stage, a special adjustment is made to training parameters at the end of each epoch. The adjustment means there will be a discontinuity in the training curve between epochs. This adjustment will eventually make the loss turn NEGATIVE. This is normal. If your training align_loss does not eventually go negative, you likely need to train more.

At each validation step, both an un-adjusted align_loss and a confidence score are generated. align_loss should be going down. Confidence should be going up. You want to pick a number of epochs so that these scores reach the knee in their curve. Do not keep training forever just because they are slowly going down. If you run into issues where things are not converging later, it is likely that you need to come back to this step and train a different amount to hit that knee in the curve of loss.

During alignment pre-training, we ALSO train on the validation set. This is usually a very very bad thing in ML. But in this case, the alignment model will never be used for aligning out-of-distribution segments. Doing this gives us a more representative sample for acoustic and textual training and does not have any other effects on overall training.

```
cd stylish-tts
uv run stylish-train train-align /path/to/your/config.yml --out /path/to/your/output
```

The `--out` option is where logs and checkpoints will end up. Once the alignment stage completes, it will provide a trained model at the file specified in your `config.yml`. It is important to realize that this is a MODEL, not the alignments themselves. We will use this model to generate the alignments.

```
cd stylish-tts
uv run stylish-train align /path/to/your/config.yml
```

This generates the actual cached alignments for all the segments for both training and validation data as configured in your config.yml. You should now add the resulting alignment.safetensors path to your config.yml.

#### OPTIONAL: Culling Bad Alignments

Running align_text.py generates a score for every segment it processes. These scores are written to files in your dataset `path`. This is a confidence value. Confidence is not a guarantee of accuracy. The model might be confidently wrong of course. But it is a safe bet that the segments it is least confident about either have a problem (perhaps the text doesn't match the audio) or are just a bad fit for the model's heuristics. Culling the segments with the least confidence will make your model converge faster, though it also means it will see less training data. I have found that culling the 10% with the lowest confidence scores is a good balance.

## Starting a training run

All of the commands above should only need to be done once per dataset as long as the dataset does not change. Once they are done, their results are kept in your dataset directory. Now we begin actually training.

Here is a typical command to start off a new training run using a single machine.

```
cd stylish-tts
uv run stylish-train train /path/to/your/config.yml --out /path/to/your/output
```

--out: This is the destination path for all checkpoints, training logs, and tensorboard data. A separate sub-directory is created for each stage of training. Make sure to have plenty of disk space available here as checkpoints can take a large amount of storage.

Training happens over the course of four stages. When you begin training, it will start with the `acoustic` stage by default. The main four stages of training are `acoustic`, `textual`, `style`, and `duration`. As each stage ends, the next will automatically begin. You can specify a stage with the `--stage` option which is necessary if you are resuming from a checkpoint. Each stage has its own logs and tensorboard data in a separate subdirectory of the out_dir.


## Expectations during training

It will take a long time to run this script. So it is a good idea to run using screen or tmux to have a persistent shell that won't disappear if you get disconnected or close the window.

Stages advance automatically and a checkpoint is created at the end of every stage before moving to the next. Other checkpoints will be saved and validations will be periodically run based on your config.yml settings. Every stage will have its own sub-directory of `out` and its own training log and tensorboard graphs/samples.

### Acoustic training

Acoustic training is about training the fundamental acoustic speech prediction models which feed into the vocoder. We 'cheat' by feeding these models parameters derived directly from the audio segments. The pitch, energy, and alignments all come from our target audio. Pitch and energy are still being trained here, but they are not being used to generate predicted audio.

The main loss figure to look at is `mel` which is a perceptual similarity of the generated audio to the ground truth. It should slowly decrease during training, but the exact point at which it converges will depend on your dataset. The other loss figures can generally be ignored and may not vary much during training.

By the end of acoustic training, the samples should sound almost identical to ground-truth. These are probably going to be the best-sounding samples you listen to. But of course this is because it is doing the easiest version of the task.

### Textual training

In textual training, the acoustic speech prediction is frozen while the focus of training becomes pitch and energy. An acoustic style model still 'cheats' by using audio to generate a prosodic style. This style along with the base text are what is used to calculate the pitch and energy values for each time location.

Here, `mel`, `pitch`, and `energy` losses are all important. You should expect mel loss to always be much higher in this stage than the acoustic stage. And it will only very gradually go down. Since there are three losses here, keeping an eye on total loss is more useful. It will be a lot less stable than in acoustic, but there is still a clear trend downwards.

As training goes on, the voice should sound less strained, less 'warbly', and more natural. Make sure you are listening for the tone of the sound and how loud it is rather than strict prosody because the samples are still using the ground truth alignment.

### Style training

Here the only 'cheating' we do is to use the ground-truth alignment. The predicted pitch and energy are used to directly predict the audio. A textual style encoder is trained to produce the same outputs as the acoustic model from the previous stage.

Aside from that, the training regimen should look a lot like the previous stage. `mel`, `pitch`, and `energy` should all trend downward but expect `mel` to be higher than the previous stage.

### Duration training

The final stage of training removes our last 'cheat' and trains the duration predictor to try to replicate the prosody of the original. The other models are frozen. All samples use only values predicted from the text.

The `duration` and `duration_ce` losses should both slowly go down. The main danger here is overfitting. So if you see validation loss stagnate or start going up you should stop training even if training loss is still going down. It is expected that one of the losses might plateau before the other.

When you listen to samples, you will get the same version you'd expect to hear during inference. Listen to make sure the voice as a whole is not going to fast or slow or just going past punctuation without pausing. You should no longer expect it to mirror the ground truth exactly, but it should have generalized to the point where it is a plausible and expressive reading. As training proceeds, it should sound more and more like fluent prosody. If there are still pitch or energy issues like warbles or loudness or tone, then those won't be fixed in this stage and you may need to train more in Textual or Acoustic before trying Duration training.

## Loading a checkpoint

```
cd stylish-tts
uv run stylish-train train \
    /path/to/your/config.yml \
    --stage <stage>
    --out /path/to/your/output \
    --checkpoint /path/to/your/checkpoint
```

You can load a checkpoint from any stage via the --checkpoint argument. You still need to set --stage appropriately to one of "alignment|acoustic|acoustic|textual|textual|duration". If you set it to the same stage as the checkpoint loaded from, it will continue in that stage at the same step number and epoch. If it is a different stage, it will train the entire stage.

Note that Stylish TTS checkpoints are not compatible with StyleTTS 2 checkpoints.

# Export to ONNX
This command will export two ONNX files, one for predicting duration and the other for predicting speech.

```sh
cd stylish-tts
uv run stylish-train convert \
    /path/to/your/config.yml \
    --duration /path/to/your/duration.onnx
    --speech /path/to/your/speech.onnx \
    --checkpoint /path/to/your/checkpoint
```

Using the ONNX model:
```sh
cd stylish-tts
uv run stylish-tts/train/test_onnx.py
    --speech /path/to/your/output/speech.onnx \
    --duration /path/to/your/output/duration.onnx \
    --text "ðˈiːz wˈɜː tˈuː hˈæv ˈæn ɪnˈɔːɹməs ˈɪmpækt , nˈɑːt ˈoʊnliː bɪkˈɔz ðˈeɪ wˈɜː əsˈoʊsiːˌeɪtᵻd wˈɪð kˈɑːnstəntˌiːn ," \
    --text "bˈʌt ˈɔlsoʊ bɪkˈɔz , ˈæz ɪn sˈoʊ mˈɛniː ˈʌðɚ ˈɛɹiːəz , ðə dɪsˈɪʒənz tˈeɪkən bˈaɪ kˈɑːnstəntˌiːn ( ˈɔːɹ ɪn hˈɪz nˈeɪm ) wˈɜː tˈuː hˈæv ɡɹˈeɪt səɡnˈɪfɪkəns fˈɔːɹ sˈɛntʃɚiːz tˈuː kˈʌm ." \
    --combine true
```

Content: These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come.

# Training New Languages

## Grapheme to Phoneme (G2P)

Grapheme-to-phoneme conversion (G2P) is the task of transducing graphemes (i.e. letter(s) that spells a sound) to phonemes (i.e. the sound).  Each language has its own phonetic rules, therefore requires a distinct G2P system. Accurate G2P is critical for the performance of text-to-speech (TTS).

The most effective G2P systems are typically tailored to specific languages. These can often be found in research papers focused on phonetics or TTS—try searching for terms like "[language] G2P/TTS site:arxiv.org" or "[language] G2P site:github.com". Libraries such as [misaki](https://github.com/hexgrad/misaki/) may also provide such G2P systems.

A commonly used multilingual G2P system is `espeak-ng`, though its accuracy can vary depending on the language. In some cases, a simple approach—using word-to-phoneme mappings from sources like Wiktionary—can be sufficient.

## Adjust model.yml

If the G2P don't share the same phonetic symbol set in `model.yml`, change symbol section and text_encoder.tokens. text_encoder.tokens should be equal to length of pad + punctuation + letters + letters_ipa
```
...
text_encoder:
  tokens: 178 # number of phoneme tokens
  hidden_dim: 192
  filter_channels: 768
  heads: 2
  layers: 6
  kernel_size: 3
  dropout: 0.1

...

symbol:
  pad: "$"
  punctuation: ";:,.!?¡¿—…\"()“” "
  letters: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  letters_ipa: "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁᵊǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

```

# Citations

Most code taken from other sources is MIT-licensed and all original code in this repository is MIT-licensed. A BSD license is included as a comment before the few pieces of code which were BSD-licensed.

- The foundation of this work is StyleTTS and StyleTTS 2 
  - "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis" by Yinghao Aaron Li, Cong Han, Nima Mesgarani [Paper](https://arxiv.org/abs/2205.15439) [Code](https://github.com/yl4579/StyleTTS)
  - "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models" by Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani [Paper](https://arxiv.org/abs/2306.07691) [Code](https://github.com/yl4579/StyleTTS2)

- Discriminators
  - "Improve GAN-based Neural Vocoder using Truncated Pointwise Relativistic Least Square GAN" by Yanli Li, Congyi Wang [Paper](https://dl.acm.org/doi/abs/10.1145/3573834.3574506)
  - Some code adapted from "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis" [Paper](https://arxiv.org/pdf/2306.00814.pdf) [Code](https://github.com/gemelo-ai/vocos)
  - Discriminator Regulator: "Mind the (optimality) Gap: A Gap-Aware Learning Rate Scheduler for
  Adversarial Nets" by Hussein Hazimeh, Natalia Ponomareva [Paper](https://arxiv.org/abs/2302.00089) [Code](https://github.com/google-research/google-research/blob/master/adversarial_nets_lr_scheduler/demo.ipynb)
  - Only use MRD discriminator: "GAN Vocoder: Multi-Resolution Discriminator Is All You Need" by Jaeseong You, Dalhyun Kim, Gyuhyeon Nam, Geumbyeol Hwang, Gyeongsu Chae [Paper](https://www.isca-archive.org/interspeech_2021/you21b_interspeech.pdf)

- Text Alignment
  - "Less Peaky and More Accurate CTC Forced Alignment by Label Priors" by Ruizhe Huang, Xiaohui Zhang, Zhaoheng Ni, Li Sun, Moto Hira, Jeff Hwang, Vimal Manohar, Vineel Pratap, Matthew Wiesner, Shinji Watanabe, Daniel Povey, Sanjeev Khudanpur [Paper](https://arxiv.org/abs/2406.02560v3) [Code](https://github.com/huangruizhe/audio/tree/aligner_label_priors/examples/asr/librispeech_alignment)
  - "Evaluating Speech–Phoneme Alignment and Its Impact on Neural Text-To-Speech Synthesis" by Frank Zalkow, Prachi Govalkar, Meinard Müller, Emanuël A. P. Habets, and Christian Dittmar [Paper](https://ieeexplore.ieee.org/document/10097248) [Supplement](https://www.audiolabs-erlangen.de/resources/NLUI/2023-ICASSP-eval-alignment-tts)
  - "Phoneme-to-Audio Alignment with Recurrent Neural Networks for Speaking and Singing Voice" by Yann Teytaut, Axel Roebel [Paper](https://www.isca-archive.org/interspeech_2021/teytaut21_interspeech.html)

- Pitch Extraction
  - "Harvest: A high-performance fundamental frequency estimator from speech signals" by Masanori Morise [Paper](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf)
  - "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music" by Haojie Wei, Xueke Cao, Tangpeng Dan, Yueguo Chen [Paper](https://arxiv.org/abs/2306.15412)

- Text Encoding
  - Taken from "Matcha-TTS: A fast TTS architecture with conditional flow matching", by Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, and Gustav Eje Henter [Paper](https://arxiv.org/abs/2309.03199) [Code](https://github.com/shivammehta25/Matcha-TTS)
  - Originally from "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search" by Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon [Paper](https://arxiv.org/abs/2005.11129) [Code](https://github.com/jaywalnut310/glow-tts)

- Vocoder is a hybrid model with inspiration from several sources
  <!-- - Backbone: "RingFormer: A Neural Vocoder with Ring Attention and Convolution-Augmented Transformer" by Seongho Hong, Yong-Hoon Choi [Paper](https://arxiv.org/abs/2501.01182) [Code](https://github.com/seongho608/RingFormer) -->
  <!-- - Harmonics Generation: "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis" by Wang, X., Takaki, S. & Yamagishi, J. [Paper](https://ieeexplore.ieee.org/document/8915761) [Code](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/tree/master/waveform-modeling/project-NSF-v2-pretrained) -->
  - "APNet2: High-quality and High-efficiency Neural Vocoder with Direct Prediction of Amplitude and Phase Spectra" by Hui-Peng Du, Ye-Xin Lu, Yang Ai, Zhen-Hua Ling [Paper](https://arxiv.org/abs/2311.11545)
  - "LightVoc: An Upsampling-Free GAN Vocoder Based On Conformer And Inverse Short-time Fourier Transform" by Dinh Son Dang, Tung Lam Nguyen, Bao Thang Ta, Tien Thanh Nguyen, Thi Ngoc Anh Nguyen, Dang Linh Le, Nhat Minh Le, Van Hai Do [Paper](https://www.isca-archive.org/interspeech_2023/dang23b_interspeech.pdf)
  - For phase loss and serial AP architecture (even though we found quality is better with discriminator and also the phase loss): "Is GAN Necessary for Mel-Spectrogram-based Neural Vocoder?" by Hui-Peng Du, Yang Ai, Rui-Chen Zheng, Ye-Xin Lu, Zhen-Hua Ling [Paper](https://arxiv.org/pdf/2508.07711)
  - For anti-wrapping phase loss: "Neural Speech Phase Prediction based on Parallel Estimation Architecture and Anti-Wrapping Losses" by Yang Ai, Zhen-Hua Ling [Paper](https://arxiv.org/abs/2211.15974)
  - Attention code from Conformer implementation by Lucidrains [Code](https://github.com/lucidrains/conformer/blob/fc70d518d3770788d17a5d9799e08d23ad19c525/conformer/conformer.py#L66)

- Duration prediction
  - Ordinal regression loss: "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation" by Gorkem Polat, Ilkay Ergenc, Haluk Tarik Kani, Yesim Ozen Alahdab, Ozlen Atug, Alptekin Temizel [Paper](https://arxiv.org/abs/2202.05167)

- ONNX Compatibility
  - Kokoro [Code](https://github.com/hexgrad/kokoro) 
  - Custom STFT Contributed to Kokoro by [Adrian Lyjak](https://github.com/adrianlyjak)
  - Loopless Duration Contributed to Kokoro by [Hexgrad](https://github.com/hexgrad)

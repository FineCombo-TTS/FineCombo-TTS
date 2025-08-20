#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Generate audio using a trained Parler-TTS model"""
import soundfile as sf
import logging
import os
import sys
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from accelerate import Accelerator
from parler_tts import ParlerTTSConfig_FACodec, ParlerTTSForConditionalGeneration
from training.arguments import ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments
from datasets import Dataset, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset
from torch.utils.data import DataLoader
from training.utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from training.data import DataCollatorParlerTTSWithPadding_Generate
import numpy as np
from tqdm import tqdm
from parler_tts.SparkTTS.sparktts.utils.audio import audio_volume_normalize
import soundfile
import soxr
from audio.stft import TacotronSTFT
from audio.tools import get_mel_from_wav
from parler_tts.naturalspeech3_facodec.ns3_codec.facodec import FACodecEncoderV2


logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    # Load feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    prompt_tokenizer = AutoTokenizer.from_pretrained(model_args.prompt_tokenizer_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir,)
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    print(last_checkpoint)
    # Load model
    config = ParlerTTSConfig_FACodec.from_pretrained(model_args.model_name_or_path)
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)

    # Prepare accelerator
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(last_checkpoint)
    sampling_rate = feature_extractor.sampling_rate

    eval_data = {
        'text':[
            'The day passed away in utter silence-night came without recurrence of the noise.',
            'The day passed away in utter silence-night came without recurrence of the noise.',
            'The day passed away in utter silence-night came without recurrence of the noise.',
        ],
        'wav_path':[
            './test_audio/gpt4o_79_angry_coral.wav',
            './test_audio/gpt4o_2147_sad_verse.wav',
            './test_audio/gpt4o_17929_happy_ballad.wav',
        ]
    }
    raw_datasets = Dataset.from_dict(eval_data)
    # Load the dataset for generation
    def pass_through_processors(prompt):
        batch = {}

        batch["prompt_input_ids"] = prompt_tokenizer(prompt.strip())["input_ids"]

        return batch

    with accelerator.local_main_process_first():
        # this is a trick to avoid to rewrite the entire audio column which takes ages
        vectorized_datasets = raw_datasets.map(
            pass_through_processors,
            remove_columns=raw_datasets.column_names,
            input_columns=['text'],
            num_proc=1,
            desc="preprocess datasets",
        )


    filter_length = data_args.filter_length
    hop_length = data_args.hop_length
    win_length = data_args.win_length
    n_mel_channels = data_args.n_mel_channels
    mel_fmin = data_args.mel_fmin
    mel_fmax = data_args.mel_fmax
    STFT = TacotronSTFT(
        filter_length,
        hop_length, 
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    )

    def get_mel(wav_path):
        batch = {}
        wav, sr = librosa.load(wav_path, sr=None)
        if sr != sampling_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
        wav = wav.astype(np.float32)
        # Compute mel-scale spectrogram and energy
        mel_spectrogram, _ = get_mel_from_wav(wav, STFT)
        batch['mel_spectrogram'] = mel_spectrogram.T
        batch['mel_length'] = mel_spectrogram.shape[1]
        return batch
    
    mel_datasets = raw_datasets.map(
        get_mel,
        remove_columns=raw_datasets.column_names,
        input_columns=['wav_path'],
        num_proc=1,
        desc="preprocess mel datasets",
    )
    vectorized_datasets = concatenate_datasets([vectorized_datasets, mel_datasets], axis=1)
    print(vectorized_datasets)


    facodec_encoder_v2_ckpt = './FaCodec/ns3_facodec_encoder_v2.bin'
    fa_encoder_v2 = FACodecEncoderV2(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_encoder_v2.load_state_dict(torch.load(facodec_encoder_v2_ckpt))
    fa_encoder_v2.eval()
    fa_encoder_v2 = fa_encoder_v2.to(accelerator.device) 

    def load_audio(
        adfile: str=None,
        audio: list = None,
        ori_sr: int=44100,
        sampling_rate: int = 16000,
    ) -> np.ndarray:
        sr = ori_sr
        if adfile:
            audio, sr = soundfile.read(adfile)
            if len(audio.shape) > 1:
                audio = audio[:, 0]

        if sampling_rate is not None and sr != sampling_rate:
            audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
            sr = sampling_rate
        
        audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        return audio

    def pass_through_processors(audio):
        batch = {}

        audios = load_audio(adfile=audio)
        audios = audios.to(accelerator.device)
        with torch.no_grad(), torch.cuda.amp.autocast(): 
            enc_out = fa_encoder_v2(audios).transpose(1,2)
        batch['facodec_enc_out'] = enc_out.squeeze(0)
        batch['enc_length'] = enc_out.size(1)
        return batch    
    
    with accelerator.local_main_process_first():
        # this is a trick to avoid to rewrite the entire audio column which takes ages
        vectorized_datasets_audio = raw_datasets.map(
            pass_through_processors,
            remove_columns=raw_datasets.column_names,
            input_columns=['wav_path'],
            num_proc=1,
            desc="preprocess facodec datasets",
        )
    vectorized_datasets = concatenate_datasets([vectorized_datasets_audio, vectorized_datasets], axis=1) 
    print(vectorized_datasets)

    def generate_step(batch, accelerator):
        gen_kwargs = {
            "do_sample": model_args.do_sample,
            "temperature": model_args.temperature,
            "max_length": model_args.max_length,
            # Because of the delayed pattern mask, generation might stop earlier because of unexpected behaviour
            # on the first tokens of the codebooks that are delayed.
            # This fix the issue.
            "min_new_tokens": model.decoder.config.num_codebooks + 1,
        }
        batch.pop("decoder_attention_mask", None)
        eval_model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
        if training_args.torch_compile:
            # if the model is compiled, we use the original model bc compile is not compatible with .generate
            eval_model = model._orig_mod

        # since we've might have loaded the weights in fp32, we have to autocast to ensure FA2 weights are in half-precision.
        # with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=(attn_implementation=="flash_attention_2"))):
        # output_audios = eval_model.generate_with_cfg(**batch, **gen_kwargs, use_cfg=True, prompt_cfg_scale=1.5)
        output_audios = eval_model.generate(**batch, **gen_kwargs, prompt_cfg_scale=1.5) # 修改text cfg
        output_audios = accelerator.pad_across_processes(output_audios, dim=1, pad_index=0)
        return output_audios

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    encoder_data_collator = DataCollatorParlerTTSWithPadding_Generate(
        prompt_tokenizer=prompt_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        audio_max_length=None,
    )

    eval_preds = []
    eval_prompts = []
    validation_dataloader = DataLoader(
        vectorized_datasets,
        collate_fn=encoder_data_collator,
        batch_size=6,
        drop_last=False,
        num_workers=1,
        pin_memory=1,
    )
    validation_dataloader = accelerator.prepare(validation_dataloader)
    model.eval()
    # generation
    for batch in tqdm(
        validation_dataloader,
        desc=f"Evaluating - Generation ...",
        position=2,
        disable=not accelerator.is_local_main_process,
    ):
        generated_audios = generate_step(batch, accelerator)
        # Gather all predictions and targets
        generated_audios, prompts = accelerator.pad_across_processes(
            (generated_audios, batch["prompt_input_ids"]), dim=1, pad_index=0
        )
        generated_audios, prompts = accelerator.gather_for_metrics(
            (generated_audios, prompts)
        )
        eval_preds.extend(generated_audios.to("cpu"))
        eval_prompts.extend(prompts.to("cpu"))
        accelerator.wait_for_everyone()
    wav_dir = os.path.join(
        './output_audio',
        f"{os.path.basename(os.path.dirname(last_checkpoint))}_{os.path.basename(last_checkpoint)}_prompt-cfg-1.5"
    )
    os.makedirs(wav_dir, exist_ok=True)
    for i, audio in enumerate(eval_preds):
        wav_file = os.path.join(wav_dir, f'audio_{i}.wav')
        print(wav_file)
        sf.write(wav_file, audio, sampling_rate) 

if __name__ == "__main__":
    main()


                    


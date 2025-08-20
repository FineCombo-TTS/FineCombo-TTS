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
import yaml
import soundfile as sf
import logging
import os
import sys
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from accelerate import Accelerator
from parler_tts import ParlerTTSConfig_FACodec, ParlerTTSForConditionalGeneration, StyleTransferParlerTTS
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
    dict2namespace,
)
from training.data import load_multiple_datasets, DataCollatorParlerTTSWithPadding, DataCollatorParlerTTSWithPadding_Stage2_Generate
import numpy as np
from tqdm import tqdm
from audio.stft import TacotronSTFT
from audio.tools import get_mel_from_wav

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ParlerTTSTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    # Load feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    prompt_tokenizer = AutoTokenizer.from_pretrained(model_args.prompt_tokenizer_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir,)
    description_tokenizer = AutoTokenizer.from_pretrained(model_args.description_tokenizer_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir,)
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    print(last_checkpoint)
    # Load model
    config = ParlerTTSConfig_FACodec.from_pretrained(model_args.model_name_or_path)

    with open('./helpers/training_configs/Template-DDPM-f4.yaml', 'r') as f:
        BBDM_config = yaml.load(f, Loader=yaml.FullLoader)
    BBDM_config = dict2namespace(BBDM_config)

    style_parlertts = StyleTransferParlerTTS(
        config=BBDM_config.model,
        parlertts_config=config,
    )
    # Prepare accelerator
    accelerator = Accelerator()
    parlertts_checkpoint = f"{last_checkpoint}/pytorch_model.bin"

    style_parlertts.load_state_dict(torch.load(parlertts_checkpoint))


    model = accelerator.prepare(style_parlertts)
    sampling_rate = feature_extractor.sampling_rate

    eval_data = {
        'prompt_wav_path':[
            './test_extractor/1472_10070_1220.62.wav',
            './test_extractor/1472_10070_1220.62.wav',
            './test_extractor/1472_10070_1220.62.wav',
            '',
            '',
            ''

        ],
        'prompt_description':[
            "Change the prosody, speed up the speech rate.",
            "Change the style, speak with a happy emotion",
            "Change the timbre to a very masculine, deep, very thick, very mature, slightly old, cool voice.",
            "Generate a voice, A man's vibrant voice conveys high energy while he speaks with a low-pitched tone at a natural speed.",
            "Generate a voice, Speaking swiftly and softly, she exudes low energy.",
            "Generate a voice, The woman's speech exudes high energy while maintaining a normal tone.",

        ],
        'target_content':[
            'the clouds covered the entire sky and obscured the highest mountain peaks worse still they steadily descended lower and lower a sign of bad weather',
            'the clouds covered the entire sky and obscured the highest mountain peaks worse still they steadily descended lower and lower a sign of bad weather',
            'the clouds covered the entire sky and obscured the highest mountain peaks worse still they steadily descended lower and lower a sign of bad weather',
            "But there was a great green cross over the pulpit, and words along the walls, and festoons upon the galleries, and great wreaths, like vast green serpents, coiled about the cold pillars.",
            "Who is been repeating all that hard stuff to you?",
            "Adam could never cease to mourn over that mystery of human sorrow which had been brought so close to him; he could never thank God for another's misery.",
        ],
    }
    raw_datasets = Dataset.from_dict(eval_data)
   
    # Load the dataset for generation
    def pass_through_processors(description, prompt):
        batch = {}

        batch["description_input_ids"] = description_tokenizer(description.strip())["input_ids"]
        batch["text_input_ids"] = prompt_tokenizer(prompt.strip())["input_ids"]

        return batch

    with accelerator.local_main_process_first():
        # this is a trick to avoid to rewrite the entire audio column which takes ages
        vectorized_datasets = raw_datasets.map(
            pass_through_processors,
            remove_columns=raw_datasets.column_names,
            input_columns=['prompt_description', 'target_content'],
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
        if wav_path == '':
            batch['mel_spectrogram'] = np.zeros((1, 80), dtype=np.float32)
            batch['mel_length'] = 1
            return batch
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
        input_columns=['prompt_wav_path'],
        num_proc=1,
        desc="preprocess mel datasets",
    )

    vectorized_datasets = concatenate_datasets([vectorized_datasets, mel_datasets], axis=1)
    print(vectorized_datasets)



    from parler_tts.naturalspeech3_facodec.ns3_codec.facodec import FACodecEncoderV2
    import soundfile
    import soxr

    facodec_encoder_v2_ckpt = './FaCodec/ns3_facodec_encoder_v2.bin'
    fa_encoder_v2 = FACodecEncoderV2(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_encoder_v2.load_state_dict(torch.load(facodec_encoder_v2_ckpt))
    fa_encoder_v2.eval()
    fa_encoder_v2 = fa_encoder_v2.to(accelerator.device)  # 将模型移至当前设备以加速处理

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
        if audio == '':
            batch['facodec_enc_out'] = np.zeros((1, 256), dtype=np.float32)
            batch['enc_length'] = 1
            return batch
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
            input_columns=['prompt_wav_path'],
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
        output_audios = eval_model.generate(**batch, **gen_kwargs, prompt_cfg_scale=2, cfm_cfg_scale=1)
        output_audios = accelerator.pad_across_processes(output_audios, dim=1, pad_index=0)
        return output_audios

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    encoder_data_collator = DataCollatorParlerTTSWithPadding_Stage2_Generate(
        prompt_tokenizer=prompt_tokenizer,
        description_tokenizer=description_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        description_max_length=data_args.max_description_token_length,
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
        './output_audio_stage2',
        f"{os.path.basename(os.path.dirname(last_checkpoint))}_{os.path.basename(last_checkpoint)}"
    )
    os.makedirs(wav_dir, exist_ok=True)
    for i, audio in enumerate(eval_preds):
        wav_file = os.path.join(wav_dir, f'audio-{i}.wav')
        print(wav_file)
        sf.write(wav_file, audio, sampling_rate)

if __name__ == "__main__":
    main()


                    


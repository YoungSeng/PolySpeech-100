import os
import glob
import numpy as np
import soundfile as sf
import shutil
import re
from tqdm import tqdm
import librosa
import time
import datetime
import scipy.signal
import random
import torch
import argparse

# Assuming these modules are in the current directory or installed
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def get_config():
    parser = argparse.ArgumentParser(description="PolySpeech-100 Evaluation for Qwen2.5-Omni")

    # 1. Core Experiment Settings
    parser.add_argument("--lang", type=str, default="all", help="Specify language, e.g., 'zho_Hans', or 'all'")
    parser.add_argument("--shots", type=int, default=0, help="Number of N-shots, 0 for Zero-shot")
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought (CoT)")

    # 2. Data Augmentation
    parser.add_argument("--augment", type=str, default="clean",
                        choices=["clean", "noise_low", "noise_high", "speed_slow", "speed_fast"],
                        help="Data augmentation mode")

    # 3. Execution Control
    parser.add_argument("--start_index", type=int, default=99, help="Starting index for evaluation")
    parser.add_argument("--count", type=int, default=-1, help="Number of cases to run, -1 for all")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--save_audio", action="store_true", help="Save stitched input audio")

    # 4. Path Configuration (Converted to relative paths for GitHub)
    parser.add_argument("--base_path", type=str, default="./Restored-PolySpeech/",
                        help="Root directory of the dataset")
    parser.add_argument("--output_path", type=str, default="./results/qwen2_5_omni/",
                        help="Directory to output results")

    args = parser.parse_args()
    return args

SILENCE_DURATION = 0.2
MAX_SAVED_AUDIO_COUNT = 10 

# Relative path for GitHub repository
ASSETS_DIR = "./assets/" 
LETTER_FILES = {
    'A': os.path.join(ASSETS_DIR, "OptionA.mp3"),
    'B': os.path.join(ASSETS_DIR, "OptionB.mp3"),
    'C': os.path.join(ASSETS_DIR, "OptionC.mp3"),
    'D': os.path.join(ASSETS_DIR, "OptionD.mp3"),
    'Question': os.path.join(ASSETS_DIR, "Question.mp3")
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Prompts
SYSTEM_PROMPT_STANDARD = """You are an expert linguist taking a multiple-choice speech comprehension test.
You will hear an audio clip containing a passage, a question, and four options (A, B, C, D).

Your task is to select the correct option based on the audio content.

CRITICAL RULES:
1. Output ONLY the single letter of the correct answer (A, B, C, or D).
2. Do NOT provide explanations, transcripts, or notes.
3. Do NOT output "I don't know" or "I cannot understand".
4. The audio may contain strong regional dialects or accents. If you are unsure, you MUST make your best guess.

Example Format:
User: [Audio Input]
Assistant: B"""

SYSTEM_PROMPT_COT = """You are an expert linguist taking a multiple-choice speech comprehension test.
You will hear an audio clip containing a passage, a question, and four options (A, B, C, D).

Your task is to:
1. Briefly summarize the key information in the audio.
2. Explain why the other options are wrong.
3. State the final correct option letter.

Format:
Analysis: [Your reasoning]
Answer: [Letter]"""

def normalize_audio(audio_data):
    if np.max(np.abs(audio_data)) > 0:
        return audio_data / np.max(np.abs(audio_data)) * 0.9
    return audio_data

def stitch_audio_files(folder_path, output_filename):
    passage_files = sorted(glob.glob(os.path.join(folder_path, "*01_passage_s*_seg00.wav")))
    q_00 = glob.glob(os.path.join(folder_path, "*02_question_00.wav"))
    a_1 = glob.glob(os.path.join(folder_path, "*03_answer_1_00.wav"))
    a_2 = glob.glob(os.path.join(folder_path, "*03_answer_2_00.wav"))
    a_3 = glob.glob(os.path.join(folder_path, "*03_answer_3_00.wav"))
    a_4 = glob.glob(os.path.join(folder_path, "*03_answer_4_00.wav"))

    target_sr = 16000
    data_list = []
    silence_chunk = np.zeros(int(target_sr * SILENCE_DURATION))

    def add_audio(path):
        try:
            d, _ = librosa.load(path, sr=target_sr)
            d = normalize_audio(d)
            data_list.append(d)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def add_silence():
        data_list.append(silence_chunk)

    if passage_files:
        for p in passage_files: add_audio(p)
        add_silence() 
    else:
        print(f"Warning: No passage files found in {folder_path}")

    if q_00:
        add_audio(LETTER_FILES['Question']) 
        add_audio(q_00[0])
        add_silence() 

    if a_1:
        add_audio(LETTER_FILES['A'])
        add_audio(a_1[0])
        add_silence()
    if a_2:
        add_audio(LETTER_FILES['B'])
        add_audio(a_2[0])
        add_silence()
    if a_3:
        add_audio(LETTER_FILES['C'])
        add_audio(a_3[0])
        add_silence()
    if a_4:
        add_audio(LETTER_FILES['D'])
        add_audio(a_4[0])

    if not data_list: return False, None

    final_audio = np.concatenate(data_list)

    if AUGMENT_MODE != 'clean':
        final_audio = augment_audio(final_audio, mode=AUGMENT_MODE, sr=target_sr)

    sf.write(output_filename, final_audio, target_sr)
    audio_duration = len(final_audio) / 16000

    return True, audio_duration

def get_ground_truth_label(folder_path):
    info_path = os.path.join(folder_path, "info.txt")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r"Correct Answer:\s*(\d+)", content)
            if match:
                ans_idx = int(match.group(1))
                return chr(ord('A') + ans_idx - 1)
    except Exception as e:
        print(f"Error reading info.txt: {e}")
    return None

def extract_index_from_folder(folder_name):
    match = re.search(r'idx_(\d+)_', folder_name)
    if match: return int(match.group(1))
    return -1

def extract_answer_smart(text):
    text = text.strip().upper()
    match = re.search(r"ANSWER\s*:\s*([A-D])", text)
    if match: return match.group(1)
    match = re.search(r"THE ANSWER IS\s?([A-D])", text)
    if match: return match.group(1)
    match = re.search(r"\b([A-D])\W*$", text)
    if match: return match.group(1)
    matches = re.findall(r"\b([A-D])\b", text)
    if matches: return matches[-1]
    return text

def augment_audio(audio_data, mode='clean', sr=16000):
    if mode == 'clean': return audio_data
    elif 'noise' in mode:
        noise_level = 0.01 if 'low' in mode else 0.05
        noise = np.random.randn(len(audio_data))
        augmented = audio_data + noise_level * noise
        max_val = np.max(np.abs(augmented))
        if max_val > 0:
            augmented = augmented / max_val * 0.9
        return augmented
    elif 'speed' in mode:
        try:
            rate = 0.8 if 'slow' in mode else 1.2
            new_len = int(len(audio_data) / rate)
            return scipy.signal.resample(audio_data, new_len)
        except Exception as e:
            print(f"Augment speed error: {e}, returning original.")
            return audio_data
    return audio_data

def run_evaluation(model, current_lang, global_start_time, global_counter, total_global_tasks):
    print(f"\n{'=' * 20} Start Processing Language: {current_lang} {'=' * 20}")
    input_dir = os.path.join(BASE_DATA_PATH, f"lang={current_lang}")
    output_dir = os.path.join(OUTPUT_BASE_PATH, f"{current_lang}")
    shot_audio_dir = os.path.join(OUTPUT_BASE_PATH, "few_shot_examples", f"{current_lang}")
    
    if N_SHOTS > 0 and not os.path.exists(shot_audio_dir):
        os.makedirs(shot_audio_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subfolders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    subfolders.sort(key=lambda x: extract_index_from_folder(x))

    if N_SHOTS > 0:
        shot_folders = subfolders[-N_SHOTS:]
        target_candidates = subfolders[:-N_SHOTS]
        print(f"Few-shot Mode: Using last {N_SHOTS} samples as prompts.")
    else:
        shot_folders = []
        target_candidates = subfolders

    history_messages = []
    if N_SHOTS > 0:
        print("Building context history from shots...")
        for shot_name in shot_folders:
            shot_path = os.path.join(input_dir, shot_name)
            shot_wav_path = os.path.join(shot_audio_dir, f"{shot_name}.wav")
            shot_gt = get_ground_truth_label(shot_path)
            if not shot_gt: continue

            if not os.path.exists(shot_wav_path):
                success, _ = stitch_audio_files(shot_path, shot_wav_path)
                if not success: continue

            if ENABLE_COT:
                assistant_response = f"Analysis: Based on the audio content, the correct option is {shot_gt}.\nAnswer: {shot_gt}"
            else:
                assistant_response = shot_gt

            history_messages.append({"role": "user", "content": shot_wav_path})
            history_messages.append({"role": "assistant", "content": assistant_response})

    valid_folders = [f for f in target_candidates if extract_index_from_folder(f) >= START_INDEX]
    if COUNT == -1:
        target_folders = valid_folders
    else:
        target_folders = valid_folders[:COUNT]

    final_todo_list = []
    skipped_count = 0

    if not OVERWRITE:
        for folder_name in target_folders:
            result_txt_path = os.path.join(output_dir, f"{folder_name}.txt")
            if os.path.exists(result_txt_path):
                skipped_count += 1
            else:
                final_todo_list.append(folder_name)
    else:
        final_todo_list = target_folders

    global_counter[0] += skipped_count
    if skipped_count > 0:
        print(f"⏩ Fast-forward: Skipped {skipped_count} existing files. Processing remaining {len(final_todo_list)} tasks.")

    session_correct = 0
    session_processed = 0
    saved_audio_counter = 0

    for idx, folder_name in enumerate(final_todo_list):
        folder_path = os.path.join(input_dir, folder_name)
        result_txt_path = os.path.join(output_dir, f"{folder_name}.txt")
        current_audio_path = os.path.join(output_dir, f"{folder_name}.wav")

        ground_truth = get_ground_truth_label(folder_path)
        success, audio_duration = stitch_audio_files(folder_path, current_audio_path)
        if not success: continue

        conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

        for msg in history_messages:
            if msg['role'] == 'user':
                conversation.append({"role": "user", "content": [{"type": "audio", "audio": msg['content']}]})
            else:
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": msg['content']}]})

        conversation.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio": current_audio_path},
                {"type": "text", "text": "Please answer the question."}
            ]
        })

        try:
            text_input = model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = model.processor(text=text_input, audio=audios, images=images, videos=videos,
                                     return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = inputs.to(model.device).to(model.dtype)

            output_ids = model.generate(
                **inputs, max_new_tokens=512 if ENABLE_COT else 100,
                use_audio_in_video=False, return_audio=False
            )
            input_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_len:]
            text_output = model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            cleaned_text = text_output.strip()

            clean_prediction = extract_answer_smart(cleaned_text.upper())
            is_correct = (clean_prediction == ground_truth)
            if is_correct: session_correct += 1
            session_processed += 1
            global_counter[0] += 1

            global_elapsed = time.time() - global_start_time
            if global_counter[0] > 0:
                avg_time_global = global_elapsed / global_counter[0]
                global_eta_str = str(datetime.timedelta(seconds=int(avg_time_global * (total_global_tasks - global_counter[0]))))
            else:
                global_eta_str = "Calculating..."

            local_acc = (session_correct / session_processed) * 100 if session_processed > 0 else 0
            status_icon = "✅" if is_correct else "❌"

            print(f"[{global_counter[0]}/{total_global_tasks}] {current_lang} - {folder_name}")
            print(f"   -> GT: {ground_truth} | Pred: {clean_prediction} {status_icon}")
            print(f"   -> Local Acc: {local_acc:.2f}% | Global ETA: {global_eta_str}")

            keep_audio_file = SAVE_INPUT_AUDIO and (saved_audio_counter < MAX_SAVED_AUDIO_COUNT)

            with open(result_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"GroundTruth: {ground_truth}\n")
                f.write(f"Prediction: {clean_prediction}\n")
                f.write(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}\n")
                f.write(f"RawOutput: {cleaned_text}\n")
                f.write(f"AudioDuration: {audio_duration:.2f}\n")
                if keep_audio_file:
                    f.write(f"AudioFile: {os.path.basename(current_audio_path)}\n")

            if not keep_audio_file and os.path.exists(current_audio_path):
                os.remove(current_audio_path)
            elif keep_audio_file:
                saved_audio_counter += 1

        except Exception as e:
            print(f"   -> Error during inference: {e}")
            if os.path.exists(current_audio_path): os.remove(current_audio_path)

if __name__ == '__main__':
    args = get_config()
    LANG = args.lang
    PROCESS_ALL_LANGS = (args.lang.lower() == 'all')
    START_INDEX = args.start_index
    COUNT = args.count
    OVERWRITE = args.overwrite
    SAVE_INPUT_AUDIO = args.save_audio
    N_SHOTS = args.shots
    ENABLE_COT = args.cot
    AUGMENT_MODE = args.augment
    BASE_DATA_PATH = args.base_path

    shot_str = f"{N_SHOTS}shot" if N_SHOTS > 0 else "Zeroshot"
    cot_str = "COT" if ENABLE_COT else "Direct"
    OUTPUT_BASE_PATH = os.path.join(args.output_path, f"{shot_str}_{AUGMENT_MODE}_{cot_str}")

    SYSTEM_PROMPT = SYSTEM_PROMPT_COT if ENABLE_COT else SYSTEM_PROMPT_STANDARD

    print("\n" + "=" * 40)
    print(f"🚀 Experiment Launching on PID: {os.getpid()}")
    print(f"📌 Experiment Tag : {shot_str} | {AUGMENT_MODE} | {cot_str}")
    print(f"📂 Data Path      : {BASE_DATA_PATH}")
    print(f"📂 Output Path    : {OUTPUT_BASE_PATH}")
    print(f"🈯 Language       : {LANG}")
    print(f"🔢 Start / Count  : {START_INDEX} / {COUNT}")
    print(f"🧠 CoT Enabled    : {ENABLE_COT}")
    print(f"🔊 Augment Mode   : {AUGMENT_MODE}")
    print("=" * 40 + "\n")

    seed_everything(42)

    print("Loading Qwen2.5-Omni model...")
    model_path = "Qwen/Qwen2.5-Omni-7B"

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model.processor = processor

    target_languages = []
    if PROCESS_ALL_LANGS:
        all_dirs = os.listdir(BASE_DATA_PATH)
        for d in all_dirs:
            if d.startswith("lang=") and os.path.isdir(os.path.join(BASE_DATA_PATH, d)):
                target_languages.append(d.split("lang=")[1])
        target_languages.sort()
    else:
        target_languages = [LANG]

    total_global_tasks = 0
    for lang in target_languages:
        lang_dir = os.path.join(BASE_DATA_PATH, f"lang={lang}")
        if not os.path.exists(lang_dir): continue
        subfolders = [f for f in os.listdir(lang_dir) if f.startswith("idx_")]
        subfolders.sort(key=lambda x: extract_index_from_folder(x))
        
        target_candidates = subfolders[:-N_SHOTS] if N_SHOTS > 0 and len(subfolders) > N_SHOTS else subfolders
        valid_count = sum(1 for f in target_candidates if extract_index_from_folder(f) >= START_INDEX)
        if COUNT != -1: valid_count = min(valid_count, COUNT)
        total_global_tasks += valid_count

    global_start_time = time.time()
    global_counter = [0] 

    for i, lang in enumerate(target_languages):
        run_evaluation(model, lang, global_start_time, global_counter, total_global_tasks)

    print(f"\nAll tasks finished. Total time: {datetime.timedelta(seconds=int(time.time() - global_start_time))}")

import os
import glob
import csv

# ================= Configuration =================

# Execution Mode:
# 1 : [Single Mode] Evaluates a specific folder defined by BASE_PATH.
# 2 : [Batch Mode] Evaluates multiple models based on MODEL_LIST and EXP_TYPE.
# 4 : [Aggregated Mode] Generates a single CSV table for all models across all languages.
MODE = 1  

# Data root path for Mode 1 (Matches the output of inference_qwen2_5_omni.py)
BASE_PATH = "./results/qwen2_5_omni/Zeroshot_clean_Direct/"

# --- Settings for Mode 2 & Mode 4 ---
ROOT_PATH = "./results/" # Global results directory

MODEL_LIST = [
    "Fun-Audio-Chat", "Gemini", "LlamaOmni2", "Mimo-Audio", "Mini-omni",
    "Moshi", "Qwen2.5Omni", "Qwen2Audio", "StepAudio2", "OpenAI",
    "Llama3.1", "Llama3.2", "Qwen2.5", "Qwen3", "Qwen3-ASR-llama-3.2",
    "Qwen3-ASR-llama3.1", "Whisper-qwen3", "Whisper-qwen2_5", "Covoaudio",
    "MiniCPM-o", "PersonaPlex_efficient", "AudioOmni_refined"
]

# Experiment Type ('base', '3shot', 'noise_low', 'cot', etc.)
EXP_TYPE = "base"

MODEL_MAP = {
    "AudioOmni_refined": "AudioOmni",
    "PersonaPlex_efficient": "PersonaPlex",
    "MiniCPM-o": "MiniCPM-o",
    "Covoaudio": "Covoaudio",
    "Fun-Audio-Chat": "Fun-Audio-Chat",
    "Gemini": "gemini",
    "LlamaOmni2": "llama_omni2",
    "Mimo-Audio": "MiMo-Audio-24kHZ",
    "Mini-omni": "mini-omni",
    "Moshi": "moshi",
    "Qwen2.5Omni":"Qwen2.5Omni",
    "Qwen2Audio": "Qwen2audio",
    "StepAudio2": "StepAudio2_refined_24khz",
    "OpenAI": "openai",
    "Llama3.1": "llama3.1",
    "Llama3.2": "llama-3.2",
    "Qwen2.5": "qwen2_5",
    "Qwen3": "qwen3",
    "Qwen3-ASR-llama-3.2": "Qwen3-ASR-llama-3.2",
    "Qwen3-ASR-llama3.1": "Qwen3-ASR-llama3.1",
    "Whisper-qwen3": "Whisper-qwen3",
    "Whisper-qwen2_5": "Whisper-qwen2_5",
}

EXP_MAP = {
    "base": "Zeroshot_clean_Direct",        # Standard Baseline
    "cot": "Zeroshot_clean_COT",            # Chain of Thought
    "3shot": "3shot_clean_Direct",          # In-context Learning
    "noise_low": "Zeroshot_noise_low_Direct",   
    "noise_high": "Zeroshot_noise_high_Direct", 
    "speed_fast": "Zeroshot_speed_fast_Direct", 
    "speed_slow": "Zeroshot_speed_slow_Direct"  
}

# Output format: 'en' (English only), 'zh' (Chinese only), 'all' (Both)
OUTPUT_MODE = 'all'

# ================= Language Mappings =================

FLORES_EN = {
    "acm_Arab": "Mesopotamian Arabic", "afr_Latn": "Afrikaans", "als_Latn": "Tosk Albanian",
    "amh_Ethi": "Amharic", "apc_Arab": "North Levantine Arabic", "arb_Arab": "Modern Standard Arabic",
    "arb_Latn": "Modern Standard Arabic (Romanized)", "ars_Arab": "Najdi Arabic", "ary_Arab": "Moroccan Arabic",
    "arz_Arab": "Egyptian Arabic", "asm_Beng": "Assamese", "azj_Latn": "North Azerbaijani",
    "bam_Latn": "Bambara", "ben_Beng": "Bengali", "ben_Latn": "Bengali (Romanized)",
    "bod_Tibt": "Standard Tibetan", "bul_Cyrl": "Bulgarian", "cat_Latn": "Catalan",
    "ceb_Latn": "Cebuano", "ces_Latn": "Czech", "ckb_Arab": "Central Kurdish",
    "dan_Latn": "Danish", "deu_Latn": "German", "ell_Grek": "Greek",
    "eng_Latn": "English", "est_Latn": "Estonian", "eus_Latn": "Basque",
    "fin_Latn": "Finnish", "fra_Latn": "French", "fuv_Latn": "Nigerian Fulfulde",
    "gaz_Latn": "West Central Oromo", "grn_Latn": "Guarani", "guj_Gujr": "Gujarati",
    "hat_Latn": "Haitian Creole", "hau_Latn": "Hausa", "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi", "hin_Latn": "Hindi (Romanized)", "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian", "hye_Armn": "Armenian", "ibo_Latn": "Igbo",
    "ilo_Latn": "Ilocano", "ind_Latn": "Indonesian", "isl_Latn": "Icelandic",
    "ita_Latn": "Italian", "jav_Latn": "Javanese", "jpn_Jpan": "Japanese",
    "kac_Latn": "Jingpho", "kan_Knda": "Kannada", "kat_Geor": "Georgian",
    "kaz_Cyrl": "Kazakh", "kea_Latn": "Kabuverdianu", "khk_Cyrl": "Halh Mongolian",
    "khm_Khmr": "Khmer", "kin_Latn": "Kinyarwanda", "kir_Cyrl": "Kyrgyz",
    "kor_Hang": "Korean", "lao_Laoo": "Lao", "lin_Latn": "Lingala",
    "lit_Latn": "Lithuanian", "lug_Latn": "Ganda", "luo_Latn": "Luo",
    "lvs_Latn": "Standard Latvian", "mal_Mlym": "Malayalam", "mar_Deva": "Marathi",
    "mkd_Cyrl": "Macedonian", "mlt_Latn": "Maltese", "mri_Latn": "Maori",
    "mya_Mymr": "Burmese", "nld_Latn": "Dutch", "nob_Latn": "Norwegian Bokmål",
    "npi_Deva": "Nepali", "npi_Latn": "Nepali (Romanized)", "nso_Latn": "Northern Sotho",
    "nya_Latn": "Nyanja", "ory_Orya": "Odia", "pan_Guru": "Eastern Panjabi",
    "pbt_Arab": "Southern Pashto", "pes_Arab": "Western Persian", "plt_Latn": "Plateau Malagasy",
    "pol_Latn": "Polish", "por_Latn": "Portuguese", "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian", "shn_Mymr": "Shan", "sin_Latn": "Sinhala (Romanized)",
    "sin_Sinh": "Sinhala", "slk_Latn": "Slovak", "slv_Latn": "Slovenian",
    "sna_Latn": "Shona", "snd_Arab": "Sindhi", "som_Latn": "Somali",
    "sot_Latn": "Southern Sotho", "spa_Latn": "Spanish", "srp_Cyrl": "Serbian",
    "ssw_Latn": "Swati", "sun_Latn": "Sundanese", "swe_Latn": "Swedish",
    "swh_Latn": "Swahili", "tam_Taml": "Tamil", "tel_Telu": "Telugu",
    "tgk_Cyrl": "Tajik", "tgl_Latn": "Tagalog", "tha_Thai": "Thai",
    "tir_Ethi": "Tigrinya", "tsn_Latn": "Tswana", "tso_Latn": "Tsonga",
    "tur_Latn": "Turkish", "ukr_Cyrl": "Ukrainian", "urd_Arab": "Urdu",
    "urd_Latn": "Urdu (Romanized)", "uzn_Latn": "Northern Uzbek", "vie_Latn": "Vietnamese",
    "war_Latn": "Waray", "wol_Latn": "Wolof", "xho_Latn": "Xhosa",
    "yor_Latn": "Yoruba", "zho_Hans": "Chinese (Simplified)", "zho_Hant": "Chinese (Traditional)",
    "zsm_Latn": "Standard Malay", "zul_Latn": "Zulu"
}

FLORES_ZH = {
    "acm_Arab": "美索不达米亚阿拉伯语", "afr_Latn": "南非荷兰语", "als_Latn": "托斯克阿尔巴尼亚语",
    "amh_Ethi": "阿姆哈拉语", "apc_Arab": "北黎凡特阿拉伯语", "arb_Arab": "现代标准阿拉伯语",
    "arb_Latn": "现代标准阿拉伯语(罗马化)", "ars_Arab": "纳吉迪阿拉伯语", "ary_Arab": "摩洛哥阿拉伯语",
    "arz_Arab": "埃及阿拉伯语", "asm_Beng": "阿萨姆语", "azj_Latn": "北阿塞拜疆语",
    "bam_Latn": "班巴拉语", "ben_Beng": "孟加拉语", "ben_Latn": "孟加拉语(罗马化)",
    "bod_Tibt": "标准藏语", "bul_Cyrl": "保加利亚语", "cat_Latn": "加泰罗尼亚语",
    "ceb_Latn": "宿务语", "ces_Latn": "捷克语", "ckb_Arab": "索拉尼库尔德语",
    "dan_Latn": "丹麦语", "deu_Latn": "德语", "ell_Grek": "希腊语",
    "eng_Latn": "英语", "est_Latn": "爱沙尼亚语", "eus_Latn": "巴斯克语",
    "fin_Latn": "芬兰语", "fra_Latn": "法语", "fuv_Latn": "尼日利亚富拉语",
    "gaz_Latn": "西中部奥罗莫语", "grn_Latn": "瓜拉尼语", "guj_Gujr": "古吉拉特语",
    "hat_Latn": "海地克里奥尔语", "hau_Latn": "豪萨语", "heb_Hebr": "希伯来语",
    "hin_Deva": "印地语", "hin_Latn": "印地语(罗马化)", "hrv_Latn": "克罗地亚语",
    "hun_Latn": "匈牙利语", "hye_Armn": "亚美尼亚语", "ibo_Latn": "伊博语",
    "ilo_Latn": "伊洛卡诺语", "ind_Latn": "印尼语", "isl_Latn": "冰岛语",
    "ita_Latn": "意大利语", "jav_Latn": "爪哇语", "jpn_Jpan": "日语",
    "kac_Latn": "景颇语", "kan_Knda": "卡纳达语", "kat_Geor": "格鲁吉亚语",
    "kaz_Cyrl": "哈萨克语", "kea_Latn": "佛得角克里奥尔语", "khk_Cyrl": "喀尔喀蒙古语",
    "khm_Khmr": "高棉语", "kin_Latn": "卢旺达语", "kir_Cyrl": "吉尔吉斯语",
    "kor_Hang": "韩语", "lao_Laoo": "老挝语", "lin_Latn": "林加拉语",
    "lit_Latn": "立陶宛语", "lug_Latn": "干达语", "luo_Latn": "卢奥语",
    "lvs_Latn": "标准拉脱维亚语", "mal_Mlym": "马拉雅拉姆语", "mar_Deva": "马拉地语",
    "mkd_Cyrl": "马其顿语", "mlt_Latn": "马耳他语", "mri_Latn": "毛利语",
    "mya_Mymr": "缅甸语", "nld_Latn": "荷兰语", "nob_Latn": "挪威博克马尔语",
    "npi_Deva": "尼泊尔语", "npi_Latn": "尼泊尔语(罗马化)", "nso_Latn": "北索托语",
    "nya_Latn": "尼扬贾语", "ory_Orya": "奥里亚语", "pan_Guru": "东旁遮普语",
    "pbt_Arab": "南普什图语", "pes_Arab": "西波斯语", "plt_Latn": "主要马达加斯加语",
    "pol_Latn": "波兰语", "por_Latn": "葡萄牙语", "ron_Latn": "罗马尼亚语",
    "rus_Cyrl": "俄语", "shn_Mymr": "掸语", "sin_Latn": "僧伽罗语(罗马化)",
    "sin_Sinh": "僧伽罗语", "slk_Latn": "斯洛伐克语", "slv_Latn": "斯洛文尼亚语",
    "sna_Latn": "绍纳语", "snd_Arab": "信德语", "som_Latn": "索马里语",
    "sot_Latn": "南索托语", "spa_Latn": "西班牙语", "srp_Cyrl": "塞尔维亚语",
    "ssw_Latn": "斯瓦蒂语", "sun_Latn": "巽他语", "swe_Latn": "瑞典语",
    "swh_Latn": "斯瓦希里语", "tam_Taml": "泰米尔语", "tel_Telu": "泰卢固语",
    "tgk_Cyrl": "塔吉克语", "tgl_Latn": "他加禄语", "tha_Thai": "泰语",
    "tir_Ethi": "提格雷尼亚语", "tsn_Latn": "茨瓦纳语", "tso_Latn": "聪加语",
    "tur_Latn": "土耳其语", "ukr_Cyrl": "乌克兰语", "urd_Arab": "乌尔都语",
    "urd_Latn": "乌尔都语(罗马化)", "uzn_Latn": "北乌兹别克语", "vie_Latn": "越南语",
    "war_Latn": "瓦腊伊语", "wol_Latn": "沃洛夫语", "xho_Latn": "科萨语",
    "yor_Latn": "约鲁巴语", "zho_Hans": "简体中文", "zho_Hant": "繁体中文",
    "zsm_Latn": "标准马来语", "zul_Latn": "祖鲁语",    "sichuan": "四川",
    "hubei": "湖北", "cantonese": "广东", "wuzhong": "吴忠",
    "shan1xi": "山西", "suhang": "苏杭", "shanghai": "上海",
    "hunan": "湖南", "shan3xi": "陕西", "minnan": "闽南",
    "henan": "河南", "shandong": "山东", "jiangxi": "江西",
    "ningxia": "宁夏", "gansu": "甘肃", "yunnan": "云南",
    "dongbei": "东北", "guizhou": "贵州", "tianjin": "天津"
}

# ================= Core Logic =================

def get_stats(base_path):
    """
    Scans the directory and calculates accuracy statistics.
    Returns: stats dictionary
    """
    stats = {}
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist.")
        return stats

    lang_dirs = sorted(os.listdir(base_path))
    print(f"Scanning {len(lang_dirs)} items in: {base_path}...")

    for lang_dir in lang_dirs:
        full_lang_path = os.path.join(base_path, lang_dir)

        if os.path.isdir(full_lang_path):
            txt_files = glob.glob(os.path.join(full_lang_path, "*.txt"))
            if not txt_files: continue

            current_correct = 0
            current_total = 0

            for file_path in txt_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        found_result = False
                        is_correct = False

                        for line in lines:
                            if line.strip().startswith("Result:"):
                                result_val = line.split(":", 1)[1].strip().upper()
                                found_result = True
                                if "INCORRECT" in result_val:
                                    is_correct = False
                                elif "CORRECT" in result_val:
                                    is_correct = True
                                else:
                                    is_correct = False
                                break

                        if found_result:
                            current_total += 1
                            if is_correct: current_correct += 1
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")

            if current_total > 0:
                stats[lang_dir] = {
                    'correct': current_correct,
                    'total': current_total,
                    'accuracy': (current_correct / current_total) * 100
                }
    return stats

def save_report(stats, base_path, save_dir, file_prefix, mode='en'):
    """
    Saves the aggregated results to CSV and TXT files.
    mode: 'en' or 'zh'
    """
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    if mode == 'zh':
        suffix = "_zh"
        mapping = FLORES_ZH
        headers_csv = ['语言代码', '语言名称', '总数', '正确数', '正确率(%)']
        headers_txt = {
            'title': "=== 评测报告 ===",
            'source': "数据源: ",
            'overall': "总体准确率: ",
            'cols': "{:<20} | {:<25} | {:<10} | {:<10} | {:<10}",
            'col_names': ('代码', '语言名称', '正确数', '总数', '准确率')
        }
    else:
        suffix = "_en"
        mapping = FLORES_EN
        headers_csv = ['Code', 'Language Name', 'Total', 'Correct', 'Accuracy(%)']
        headers_txt = {
            'title': "=== EVALUATION REPORT ===",
            'source': "Source: ",
            'overall': "OVERALL ACCURACY: ",
            'cols': "{:<20} | {:<25} | {:<10} | {:<10} | {:<10}",
            'col_names': ('Code', 'Language Name', 'Correct', 'Total', 'Accuracy')
        }

    output_csv = os.path.join(save_dir, f"{file_prefix}_summary{suffix}.csv")
    output_txt = os.path.join(save_dir, f"{file_prefix}_report{suffix}.txt")

    # 1. Save CSV
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers_csv)

        for lang_code, data in stats.items():
            lang_name = mapping.get(lang_code, lang_code)
            writer.writerow([
                lang_code, lang_name, data['total'], data['correct'], f"{data['accuracy']:.2f}"
            ])

    # 2. Save TXT
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"{headers_txt['title']}\n")
        f.write(f"{headers_txt['source']}{base_path}\n")
        f.write("=" * 80 + "\n\n")

        total_correct = sum(d['correct'] for d in stats.values())
        total_count = sum(d['total'] for d in stats.values())
        if total_count > 0:
            avg_acc = (total_correct / total_count) * 100
            f.write(f"{headers_txt['overall']}{avg_acc:.2f}% ({total_correct}/{total_count})\n\n")

        f.write(headers_txt['cols'].format(*headers_txt['col_names']) + "\n")
        f.write("-" * 85 + "\n")

        for lang_code, data in stats.items():
            lang_name = mapping.get(lang_code, lang_code)
            if len(lang_name) > 23: lang_name = lang_name[:20] + "..."

            f.write(headers_txt['cols'].format(
                lang_code, lang_name, str(data['correct']),
                str(data['total']), f"{data['accuracy']:.2f}%"
            ) + "\n")

    print(f"[{mode.upper()}] Results saved to:\n  - {output_csv}\n  - {output_txt}")

def run_task(data_path, save_path, prefix):
    """Helper function to execute a single evaluation parsing task"""
    print(f"Processing: {prefix} ...")
    print(f"  -> Reading from: {data_path}")

    stats = get_stats(data_path)

    if not stats:
        print(f"  -> [Warning] No results found for {prefix}. Skipping.")
        return

    if OUTPUT_MODE == 'all':
        save_report(stats, data_path, save_path, prefix, 'en')
        save_report(stats, data_path, save_path, prefix, 'zh')
    elif OUTPUT_MODE == 'zh':
        save_report(stats, data_path, save_path, prefix, 'zh')
    else:
        save_report(stats, data_path, save_path, prefix, 'en')
    print(f"  -> Done.")

def save_aggregated_excel(all_data_map, all_languages, save_dir):
    """(Mode 4) Saves an aggregated CSV containing all models."""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"Aggregated_Results_{EXP_TYPE}.csv"
    output_path = os.path.join(save_dir, filename)

    header = ["Code", "Language"] + MODEL_LIST
    print(f"Saving aggregated report to: {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for lang_code in sorted(list(all_languages)):
            lang_name = FLORES_EN.get(lang_code, FLORES_EN.get(lang_code, lang_code))
            row = [lang_code, lang_name]

            for model_name in MODEL_LIST:
                if model_name in all_data_map:
                    if lang_code in all_data_map[model_name]:
                        acc = all_data_map[model_name][lang_code]['accuracy']
                        row.append(f"{acc:.2f}")
                    else:
                        row.append("-")
                else:
                    row.append("N/A")

            writer.writerow(row)
    print("Done.")

def main():
    if MODE == 1:
        # === Mode 1: Single Path Mode ===
        if not os.path.exists(BASE_PATH):
            print(f"Error: Path {BASE_PATH} does not exist.")
            return
        
        # Will save the summary next to the folders inside BASE_PATH
        run_task(BASE_PATH, BASE_PATH, "evaluation")

    elif MODE == 2:
        # === Mode 2: Batch Processing Mode ===
        save_root = os.path.join(os.getcwd(), "evaluation_report")
        print(f"Batch Mode: {len(MODEL_LIST)} models configured.")
        print(f"Experiment Type: {EXP_TYPE} ({EXP_MAP.get(EXP_TYPE, 'Unknown')})")
        print(f"Results will be saved to: {save_root}\n")

        for model_name in MODEL_LIST:
            if model_name not in MODEL_MAP:
                print(f"[Skip] Model '{model_name}' not defined in MODEL_MAP.")
                continue

            sub_folder = MODEL_MAP[model_name]
            if EXP_TYPE not in EXP_MAP:
                print(f"[Error] Experiment '{EXP_TYPE}' not defined in EXP_MAP.")
                break 

            exp_folder = EXP_MAP[EXP_TYPE]
            full_data_path = os.path.join(ROOT_PATH, sub_folder, exp_folder)
            file_prefix = f"{model_name}_{EXP_TYPE}"

            run_task(full_data_path, save_root, file_prefix)
            print("-" * 40)
            
    elif MODE == 4:
        # === Mode 4: Aggregated Report Mode ===
        save_root = os.getcwd()
        print("=== Mode 4: Aggregating all results into one file ===")
        print(f"Experiment Type: {EXP_TYPE}")

        all_models_data = {}
        all_detected_languages = set()

        for model_name in MODEL_LIST:
            print(f"Reading data for: {model_name}...")
            if model_name not in MODEL_MAP:
                print(f"  -> [Warning] Model '{model_name}' not in MODEL_MAP. Skipping.")
                continue

            sub_folder = MODEL_MAP[model_name]
            if EXP_TYPE not in EXP_MAP:
                print(f"  -> [Error] Exp '{EXP_TYPE}' not in EXP_MAP.")
                break
            
            exp_folder = EXP_MAP[EXP_TYPE]
            full_data_path = os.path.join(ROOT_PATH, sub_folder, exp_folder)

            stats = get_stats(full_data_path)
            if stats:
                all_models_data[model_name] = stats
                for lang in stats.keys():
                    all_detected_languages.add(lang)
            else:
                print(f"  -> No data found.")
                all_models_data[model_name] = {}

        if not all_detected_languages:
            print("No data found for any model.")
            return

        save_aggregated_excel(all_models_data, all_detected_languages, save_root)

    print("\nAll evaluation tasks completed.")

if __name__ == "__main__":
    main()
import yaml
import subprocess
import os
import sys

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guide-file", type=str, default="benchmark_guide.yml")
    parser.add_argument("--benchmark-script", type=str, default="benchmark.sh")
    args = parser.parse_args()

    # 파일 경로 설정 (현재 스크립트 위치 기준)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    guide_path = os.path.join(script_dir, args.guide_file)
    benchmark_script = os.path.join(script_dir, args.benchmark_script)    

    if not os.path.exists(guide_path):
        print(f"Error: Guide file not found at {guide_path}")
        sys.exit(1)

    print(f"Loading config from {guide_path}...")
    with open(guide_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)

    if not config:
        print("Config is empty")
        return

    for model_name, settings_list in config.items():
        print(f"\nProcessing model: {model_name}")
        
        # 리스트 형태의 설정을 하나의 딕셔너리로 변환
        settings = {}
        if isinstance(settings_list, list):
            for item in settings_list:
                if isinstance(item, dict):
                    settings.update(item)
        elif isinstance(settings_list, dict):
            settings = settings_list
        else:
            print(f"Warning: Settings for {model_name} is not a valid format, skipping...")
            continue
            
        # 파라미터 추출 (기본값 설정 가능)
        model_id = settings.get('model')
        tp_size = settings.get('tp_size', 1)
        dp_size = settings.get('dp_size', 1)
        pp_size = settings.get('pp_size', 1)
        enable_ep = settings.get('enable_ep', 0)
        rsd_size = settings.get('rsd_size', 1)
        max_seq_len = settings.get('max_seq_len', 4096)
        block_size = settings.get('block_size', 4096)
        length = settings.get('len', 2048)
        duration = settings.get('duration', 1800)
        max_num_sequences_list = settings.get('max_num_sequences')
        platform = settings.get('platform', 'torch_compile')
        
        if not platform in ['torch_compile', 'optimum']:
            print(f"Skipping {model_name}: 'platform' parameter is not valid, skipping...")
            continue

        if not model_id:
            print(f"Skipping {model_name}: 'model' parameter missing")
            continue

        if max_num_sequences_list is None:
            print(f"Skipping {model_name}: 'max_num_sequences' parameter missing")
            continue

        if not isinstance(max_num_sequences_list, list):
            max_num_sequences_list = [max_num_sequences_list]
            
        for max_num_seqs in max_num_sequences_list:
            print(f"--> Running configuration: max_num_seqs={max_num_seqs}, length={length}")
            
            cmd = [
                benchmark_script,
                "--platform", str(platform),
                "--model-id", str(model_id),
                "--tp-size", str(tp_size),
                "--dp-size", str(dp_size),
                "--pp-size", str(pp_size),
                "--enable-ep", str(enable_ep),
                "--rsd-size", str(rsd_size),
                "--max-seq-len", str(max_seq_len),
                "--block-size", str(block_size),
                "--length", str(length),
                "--max-num-seqs", str(max_num_seqs),
                "--duration", str(duration)
            ]
            print(f"Running command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! Benchmark failed for {model_name} (max_num_seqs={max_num_seqs}) with exit code {e.returncode}")
                # 계속 진행할지 여부 결정 (여기서는 계속 진행)
                continue
            except Exception as e:
                print(f"!!! An error occurred while running benchmark: {e}")
                continue

if __name__ == "__main__":
    main()


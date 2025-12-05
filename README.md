# guidellm-rbln-vllm

Tools for benchmarking vLLM performance with RBLN backend support.

## Components Overview

### Core Scripts
- **`running_benchmarks.py`**: The main automation script. It parses the YAML configuration and sequentially executes benchmarks by invoking `benchmark.sh` for each scenario.
- **`benchmark.sh`**: The execution script that handles the lifecycle of a single benchmark run. It:
  1. Starts the inference vLLM server (torch_compile or optimum).
  2. Verifies server health.
  3. Executes the `guidellm` benchmark command.
  4. Performs cleanup (process termination) after completion.

### Configuration
- **`benchmark_guide.yml`**: A YAML file used to define multiple benchmark scenarios. Key parameters include:
  - `model`: Hugging Face model ID.
  - `platform`: Backend to use (`torch_compile` or `optimum`).
  - `max_num_sequences`: List of concurrency levels (batch sizes) to test.
  - `tp_size`: Tensor parallelism size.
  - `max_seq_len`: Maximum sequence length.
  - `block_size`: Block size for PagedAttention.
  - `len`: Input/Output token length.
  - `duration`: Duration of the benchmark run.

## Usage

### 1. Run Benchmark

1. Configure your scenarios in `benchmark_guide.yml`.
2. Execute the runner script:

```bash
python running_benchmarks.py
# Options:
# python running_benchmarks.py --guide-file custom_config.yml --benchmark-script ./custom_benchmark.sh
```

### 2. Analyze Results

Use `extract_metrics.py` to parse the generated CSV output and display a summary of key performance metrics (Throughput, TTFT, TPOT, ITL).

```bash
python extract_metrics.py path/to/results/benchmarks.csv
```

**Example Output:**
```text
Output Throughput (Mean) , TTFT (Median) , TTFT (Mean) , TTFT (Max) , TPOT (Median) , TPOT (Mean) , TPOT (Max) , ITL (Median) , ITL (Mean) , ITL (Max)
115.973                  , 238.557       , 239.008     , 255.759    , 8.622         , 8.626       , 8.692      , 8.511        , 8.513      , 8.579    
```

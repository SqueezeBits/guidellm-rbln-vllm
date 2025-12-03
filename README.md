# guidellm-rbln-vllm

Tools for benchmarking vLLM performance with RBLN backend support.

## Usage

### 1. Run Benchmark

1. Configure your benchmark scenarios in `benchmark_guide.yml`.
2. Run the benchmark script:

```bash
python running_benchmarks.py
# Or with a custom config file:
# python running_benchmarks.py --guide-file your_config.yml
```

### 2. Analyze Results

Extract key metrics from the generated CSV results:

```bash
python extract_metrics.py results/your/benchmark/result.csv
```

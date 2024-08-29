# vLLM Benchmark

This repository, based on vLLM Benchmark, provides NVIDA LLM NIM users to run performance testing against NIMs. This will do everything vLLM Benchmark does. It NIM support has been bolted on. This repo is not affiliated with NVIDIA.

## Features

- Added support for NIVDIA LLM NIMs
- Benchmark NVIDIA LLM NIMs with different concurrency levels
- Measure key performance metrics:
  - Requests per second
  - Latency
  - Tokens per second
  - Time to first token
- Easy to run with customizable parameters
- Generates JSON output for further analysis or visualization

## Requirements

- Python 3.7+
- `openai` Python package
- `numpy` Python package

## Installation

1. Clone this repository:
   ```
   git clone (https://github.com/staggeredsix/vllm-benchmark_NIM.git)
   cd vllm-benchmark
   ```

2. Install the required packages:
   ```
   pip install openai numpy
   ```

## Usage
   ```sh ./nim-vllm-benchmark.py```
Select the test to run.
The model will load, script will detect the correct model name for requests.
Built in fun bug will have you select test again.
Manual test will allow setting requests and concurrency.
Auto Test will start with a medium load and adjust the load until returned tokens per second drops below 12.



## Output
The benchmark will report the current in-flight requests, current average TPS every 10 seconds.
After the benchmark ends it will report total concurrent requests and TPS.

Auto Test will hammer the model with requests and update you every 10 seconds with in-flight requests and TPS.
After the Auto Test detects a drop to or below 12 TPS return rate it will end the test and report maximum concurrent requests.
Rerun a few times for confirmation.

A800 LLama 3 8B Instruct was around 313 concurrent requests. Running 5 times with no specific cool down for GPU temps.

The benchmark results are saved in JSON format, containing detailed metrics for each run, including:

- Total requests and successful requests
- Requests per second
- Total output tokens
- Latency (average, p50, p95, p99)
- Tokens per second (average, p50, p95, p99)
- Time to first token (average, p50, p95, p99)

## Results

Results are dumped into the same directory as the nim-vllm-benchmark.py.

## Contributing

Contributions to improve the benchmarking scripts or add new features are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## MY MODIFICATIONS

This repo is for me to work on personal projects without breaking the original while allowing anyone to benefit from the work. Any modifications I create are freely available. I won't push any changes back to the original work as I don't trust myself to not break things.

## PENDING ENHANCEMENTS

Naming the run for logging, moving logging to results directory. Overall bug squashing. Benchmark chart creation.

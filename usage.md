# Usage

Requires [uv](https://docs.astral.sh/uv/).

## Quick Start

```bash
# All 400 questions, text-only (default: --image-mode blind)
uv run benchmark.py --model gpt-5.2 --api-key $OPENAI_API_KEY

# With clinical images via Vision API
uv run benchmark.py --model gpt-5.2 --api-key $OPENAI_API_KEY --image-mode vision

# Text-only questions only (skip 98 image questions)
uv run benchmark.py --model gpt-5.2 --api-key $OPENAI_API_KEY --image-mode skip

# OpenAI-compatible server (vLLM, etc.)
uv run benchmark.py --model my-model --base-url http://localhost:8000/v1 --api-key dummy --parallel 32

# OpenRouter (with reasoning)
uv run benchmark.py --model openai/gpt-5.2 \
  --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
  --image-mode vision --extra-body '{"reasoning": {"enabled": true}}'
```

Results are saved to `results/{model}_{timestamp}.json`. To retry failed questions:

```bash
uv run benchmark.py --model gpt-5.2 --api-key $OPENAI_API_KEY \
  --resume results/gpt-5.2_20260225_120000.json
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Model name |
| `--base-url` | `https://api.openai.com/v1` | API endpoint |
| `--api-key` | (required) | API key |
| `--parallel` | `8` | Concurrent requests |
| `--timeout` | `300` | API timeout in seconds |
| `--image-mode` | `blind` | `skip`: exclude image questions, `blind`: answer without images, `vision`: send images |
| `--extra-body` | none | JSON string passed as `extra_body` to the API |
| `--resume` | none | Resume from a previous result JSON (retry errors only) |
| `--blocks` | all | Comma-separated block filter (e.g. `A,B`) |
| `--out` | auto | Output JSON path |
| `--display-name` | same as `--model` | Display name for leaderboard |

## Reproducing Leaderboard Results

Commands used to generate each leaderboard entry.

### API Models (via OpenRouter)

- [GPT-5.2](https://openai.com/index/introducing-gpt-5-2/)
- [Gemini 3.1 Pro Preview](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/)
- [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) / [Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)


```bash
# GPT-5.2
uv run benchmark.py --model openai/gpt-5.2 --display-name "GPT-5.2" \
  --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
  --image-mode vision --extra-body '{"reasoning": {"enabled": true}}'

# Gemini 3.1 Pro Preview
uv run benchmark.py --model google/gemini-3.1-pro-preview --display-name "Gemini 3.1 Pro Preview" \
  --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
  --image-mode vision --extra-body '{"reasoning": {"enabled": true}}'

# Claude Opus 4.6
uv run benchmark.py --model anthropic/claude-opus-4.6 --display-name "Claude Opus 4.6" \
  --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
  --image-mode vision --extra-body '{"reasoning": {"enabled": true}}'

# Claude Sonnet 4.6
uv run benchmark.py --model anthropic/claude-sonnet-4.6 --display-name "Claude Sonnet 4.6" \
  --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
  --image-mode vision --extra-body '{"reasoning": {"enabled": true}}'

# Qwen3.5-397B-A17B (Alibaba provider only)
uv run benchmark.py --model qwen/qwen3.5-397b-a17b --display-name "Qwen3.5-397B-A17B" \
  --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY \
  --image-mode vision \
  --extra-body '{"provider": {"order": ["Alibaba"], "allow_fallbacks": false}, "reasoning": {"enabled": true}}'
```

### Local Models (self-hosted)

Common setup:

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

#### [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

Hardware: 4x A100 80GB (DP=4).

```bash
uv pip install vllm --torch-backend=auto

vllm serve openai/gpt-oss-20b --port 8000 \
  --data-parallel-size 4 --async-scheduling

uv run benchmark.py --model openai/gpt-oss-20b \
  --display-name "gpt-oss-20b (low)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode blind --extra-body '{"reasoning_effort": "low"}'

uv run benchmark.py --model openai/gpt-oss-20b \
  --display-name "gpt-oss-20b (medium)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode blind --extra-body '{"reasoning_effort": "medium"}'

uv run benchmark.py --model openai/gpt-oss-20b \
  --display-name "gpt-oss-20b (high)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode blind --extra-body '{"reasoning_effort": "high"}'
```

#### [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)

Hardware: 4x A100 80GB (TP=4).

```bash
uv pip install vllm --torch-backend=auto

vllm serve openai/gpt-oss-120b --port 8000 \
  --tensor-parallel-size 4 --async-scheduling

uv run benchmark.py --model openai/gpt-oss-120b \
  --display-name "gpt-oss-120b (low)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode blind --extra-body '{"reasoning_effort": "low"}'

uv run benchmark.py --model openai/gpt-oss-120b \
  --display-name "gpt-oss-120b (medium)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode blind --extra-body '{"reasoning_effort": "medium"}'

uv run benchmark.py --model openai/gpt-oss-120b \
  --display-name "gpt-oss-120b (high)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode blind --extra-body '{"reasoning_effort": "high"}'
```

#### [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)

Hardware: 4x A100 80GB (TP=4).

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-27B \
  --port 8000 --tp-size 4 \
  --mem-fraction-static 0.8 --context-length 262144 \
  --reasoning-parser qwen3

# think mode (default)
uv run benchmark.py --model Qwen/Qwen3.5-27B \
  --display-name "Qwen3.5-27B" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 8 \
  --image-mode vision

# no-think mode
uv run benchmark.py --model Qwen/Qwen3.5-27B \
  --display-name "Qwen3.5-27B (no-think)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 8 \
  --image-mode vision \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}'
```

#### [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

Hardware: 4x A100 80GB (TP=4).

```bash
uv pip install 'sglang[all]' --torch-backend=auto

python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-35B-A3B \
  --port 8000 --tp-size 4 \
  --mem-fraction-static 0.8 --context-length 262144 \
  --reasoning-parser qwen3

# think mode (default)
uv run benchmark.py --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode vision

# no-think mode (sampling params per HF recommendation)
uv run benchmark.py --model Qwen/Qwen3.5-35B-A3B \
  --display-name "Qwen3.5-35B-A3B (no-think)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode vision \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}'
```

#### [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)

Hardware: 4x A100 80GB (TP=4).

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-122B-A10B \
  --port 8000 --tp-size 4 \
  --mem-fraction-static 0.8 --context-length 262144 \
  --reasoning-parser qwen3

# think mode (default)
uv run benchmark.py --model Qwen/Qwen3.5-122B-A10B \
  --display-name "Qwen3.5-122B-A10B" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode vision

# no-think mode
uv run benchmark.py --model Qwen/Qwen3.5-122B-A10B \
  --display-name "Qwen3.5-122B-A10B (no-think)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 32 \
  --image-mode vision \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}'
```

#### [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

Hardware: 4x A100 80GB (TP=4).

```bash
vllm serve Qwen/Qwen3-32B --port 8000 \
  --tensor-parallel-size 4 --async-scheduling \
  --reasoning-parser qwen3

# think mode (default)
uv run benchmark.py --model Qwen/Qwen3-32B \
  --display-name "Qwen3-32B" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 8 \
  --image-mode blind

# no-think mode
uv run benchmark.py --model Qwen/Qwen3-32B \
  --display-name "Qwen3-32B (no-think)" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 8 \
  --image-mode blind \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}'
```

#### [GPT-OSS-Swallow-120B-RL-v0.1](https://huggingface.co/tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1)

Hardware: 4x A100 80GB (TP=4).

```bash
vllm serve tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1 --port 8000 \
  --tensor-parallel-size 4 --async-scheduling \

uv run benchmark.py --model tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1 \
  --display-name "GPT-OSS-Swallow-120B-RL-v0.1" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 8 \
  --image-mode blind
```

#### [Qwen3-Swallow-32B-RL-v0.2](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2)

Hardware: 4x A100 80GB (TP=4).

```bash
vllm serve tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2 --port 8000 \
  --tensor-parallel-size 4 --async-scheduling \
  --reasoning-parser qwen3

uv run benchmark.py --model tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2 \
  --display-name "Qwen3-Swallow-32B-RL-v0.2" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 8 \
  --image-mode blind
```

#### [Preferred-MedRECT-32B](https://huggingface.co/pfnet/Preferred-MedRECT-32B)

Hardware: 4x A100 80GB (TP=4).

```bash
vllm serve pfnet/Preferred-MedRECT-32B --port 8000 \
  --tensor-parallel-size 4 --async-scheduling \
  --reasoning-parser qwen3

uv run benchmark.py --model pfnet/Preferred-MedRECT-32B \
  --display-name "Preferred-MedRECT-32B" \
  --base-url http://localhost:8000/v1 --api-key dummy --parallel 8 \
  --image-mode blind
```

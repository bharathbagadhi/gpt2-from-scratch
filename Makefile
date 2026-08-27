.PHONY: help install data test test-all lint format train-nano train-124m finetune sample bench clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## install the package plus dev extras
	pip install -e ".[dev,pretrained]"

data:  ## download + tokenise TinyShakespeare (BPE and char)
	python scripts/prepare_data.py --dataset tinyshakespeare
	python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char

test:  ## fast test suite (no weight downloads)
	pytest -v -m "not slow"

test-all:  ## everything, including HuggingFace parity (downloads ~500MB)
	pytest -v

lint:  ## ruff check
	ruff check src tests scripts

format:  ## ruff format + autofix
	ruff format src tests scripts && ruff check --fix src tests scripts

train-nano:  ## 2-minute CPU/laptop run to prove the loop works
	python scripts/train.py --preset gpt2-nano \
		--data_dir data/tinyshakespeare_char --vocab_size 128 \
		--batch_size 8 --block_size 64 --max_steps 500 --warmup_steps 50 \
		--eval_interval 100 --eval_iters 20 --learning_rate 3e-3

train-124m:  ## full GPT-2 124M on one GPU (Colab T4 friendly)
	python scripts/train.py --preset gpt2 \
		--data_dir data/tinyshakespeare \
		--batch_size 4 --block_size 512 --total_batch_size 65536 \
		--max_steps 3000 --warmup_steps 200 --learning_rate 6e-4 \
		--eval_interval 250 --sample_interval 500

finetune:  ## fine-tune OpenAI's GPT-2 on Shakespeare
	python scripts/train.py --init_from gpt2 \
		--data_dir data/tinyshakespeare \
		--batch_size 2 --block_size 256 --grad_accum_steps 8 \
		--learning_rate 3e-5 --warmup_steps 50 --max_steps 500 \
		--dropout 0.1 --out_dir out/finetune

sample:  ## generate from the last checkpoint
	python scripts/sample.py --ckpt out/ckpt_final.pt --prompt "ROMEO:" --num_samples 2

bench:  ## throughput + memory for the current device
	python scripts/benchmark.py --preset gpt2 --batch_size 4 --block_size 512

clean:  ## remove caches and run artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage out/ __pycache__
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

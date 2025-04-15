<h1 align="center">Revela: Dense Retriever Learning via Self-supervised LM Training</h1>

<h4 align="center">
    <p>
        <a href="">📑 Paper</a> |
        <a href="#installation">🔧 Installation</a> |
        <a href="#resources">📚 Resources</a> |
        <a href="#training">🚀 Training</a> |
        <a href="#eval"> 📊 Evaluation</a> |
        <a href="#citing">📄 Citing</a>
    </p>
</h4>

> **Abstract:**
>
> Retrievers, represented by dense retrievers, play a vital role in accessing external and specialized knowledge to augment LMs.
Training dense retrievers typically requires annotated query-document pairs, which are costly and hard to scale—motivating growing interest in self-supervised solutions. However, current self-supervised approaches—such as contrastive learning, masked autoencoding, and LM-based distillation—suffer from key limitations, including oversimplified assumptions, lack of explicit negatives, or reliance on frozen LMs.
>
> In this work, we propose <code>Revela</code>, a unified and scalable framework for self-supervised retriever learning via joint retriever-LM training.
<code>Revela</code> models semantic dependencies among passages by conditioning generation on both local and cross-passage context through a novel in-batch attention mechanism.
This attention is weighted by retriever-computed similarity scores, enabling the retriever to be optimized as part of language modeling.
We evaluate <code>Revela</code> on both general-domain (BEIR) and domain-specific (CoIR) benchmarks across various retriever backbones.
With comparable parameter scales, it outperforms REPLUG by relative margins of 18.3\% and 14.4\%, respectively.
Performance scales with model size, demonstrating the effectiveness and scalability of our approach and underscoring its potential as a promising solution for self-supervised retriever learning.

<h2 id="installation">Installation</h2>

To begin, set up the conda environment using the following command:

```
conda env create -f environment.yml
```

In <code>Revela</code>, we modify the transformers architecture to incorporate **in-batch** attention. To enable this, install a customized version of the `transformers` library:

```
pip uninstall transformers
pip install git+https://github.com/TRUMANCFY/transformers.git@adapt
```

Finally, we train the model in a modular setup. To install the local package in editable mode, run:

```
cd src/tevatron
pip install -e .
```

<h2 id="resources">Resources</h2>

### Data


| Dataset                    | Source                                                                                                                                                              | Number of Batches | Batch Size |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|------------|
| [Revela Training Corpus](https://huggingface.co/datasets/trumancai/revela_training_corpus)     | [Wikipedia](https://huggingface.co/datasets/Tevatron/wikipedia-nq-corpus)                                                                                           | 320,000           | 16         |
| [Revela Code Training Corpus](https://huggingface.co/datasets/trumancai/revela_code_training_corpus) | [Stackoverflow Posts](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts), [Online Tutorials](https://huggingface.co/datasets/code-rag-bench/online-tutorials), [Library Documentation](https://huggingface.co/datasets/code-rag-bench/library-documentation) | 358,763           | 16         |


### Models

| Model Name    | Base Model                                                                 | Training Source |
|---------------|----------------------------------------------------------------------------|------------------|
| [Revela-1b](https://huggingface.co/trumancai/Revela-1b)     | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)   | [Wikipedia](https://huggingface.co/datasets/Tevatron/wikipedia-nq-corpus)        | 
| [Revela-500m](https://huggingface.co/trumancai/Revela-500M) | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)               | [Wikipedia](https://huggingface.co/datasets/Tevatron/wikipedia-nq-corpus)        |
|  [Revela-135m](https://huggingface.co/trumancai/Revela-135M) | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | [Wikipedia](https://huggingface.co/datasets/Tevatron/wikipedia-nq-corpus)    |
|  [Revela-code-1b](https://huggingface.co/trumancai/Revela-1b)     | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)   | [Stackoverflow Posts](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts) + [Online Tutorials](https://huggingface.co/datasets/code-rag-bench/online-tutorials) + [Library Documentation](https://huggingface.co/datasets/code-rag-bench/library-documentation)        |
| [Revela-code-500m](https://huggingface.co/trumancai/Revela-500M) | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)               | [Stackoverflow Posts](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts) + [Online Tutorials](https://huggingface.co/datasets/code-rag-bench/online-tutorials) + [Library Documentation](https://huggingface.co/datasets/code-rag-bench/library-documentation)        |
| [Revela-code-135m](https://huggingface.co/trumancai/Revela-code-135M) | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | [Stackoverflow Posts](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts) + [Online Tutorials](https://huggingface.co/datasets/code-rag-bench/online-tutorials) + [Library Documentation](https://huggingface.co/datasets/code-rag-bench/library-documentation)    |


<h2 id="training">Training</h2>
The training script can be found at `train.sh` under DeepSpeed training framework.


```
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRITON_PRINT_AUTOTUNING=1

export ROOT_DIR=./
export OUTPUT_DIR=...
export RUN_NAME=...

deepspeed --include localhost:0,1,2,3 --master_port 6022 --module tevatron.llm_retriever.driver.train \
  --deepspeed $ROOT_DIR/deepspeed/ds_zero3_config.json \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --lora \
  --lora_r 256 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 500 \
  --bm25_retrieval_file $DATA_PATH \
  --add_passage_prefix True \
  --add_query_prefix True \
  --first_half True \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --attn_temperature 0.0001 \
  --per_device_train_batch_size 1 \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --passage_max_len 157 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --resume latest \
  --top_k 16 \
  --run_name $RUN_NAME
```

<h2 id="eval">Evaluation</h2>

We can evaluate the trained models with customized `mteb`.

```
from mteb.model_meta import ModelMeta
from mteb.models.repllama_models import RepLLaMAWrapper, _loader

revela_llama_1b = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-3.2-1B",
        peft_model_name_or_path="trumancai/Revela-1b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="trumancai/Revela-1b",
    languages=["eng_Latn"],
    open_source=True,
    revision="41a2bd8968d2640e1e386861776c48bdaac1306a",  # base-peft revision
    release_date="2024-09-15",
)
revela_llama_1b_model = revela_llama_1b.loader()

evaluation = mteb.MTEB(tasks=["SciFact", "NFCorpus"])
evaluation.run(model=revela_llama_1b_model, output_folder="results/Revela-1b")
```

<p align="center">
  <img src="assets/beir.png" alt="" width="700"/>
</p>


<p align="center">
  <img src="assets/coir.png" alt="" width="700"/>
</p>


<h2 id="citing">Citing</h2>

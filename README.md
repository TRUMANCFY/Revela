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
> In this work, we propose $\texttt{Revela}$, a unified and scalable framework for self-supervised retriever learning via joint retriever-LM training.
$\texttt{Revela}$ models semantic dependencies among passages by conditioning generation on both local and cross-passage context through a novel in-batch attention mechanism.
This attention is weighted by retriever-computed similarity scores, enabling the retriever to be optimized as part of language modeling.
We evaluate $\texttt{Revela}$ on both general-domain (BEIR) and domain-specific (CoIR) benchmarks across various retriever backbones.
With comparable parameter scales, it outperforms REPLUG by relative margins of 18.3\% and 14.4\%, respectively.
Performance scales with model size, demonstrating the effectiveness and scalability of our approach and underscoring its potential as a promising solution for self-supervised retriever learning.


<h2 id="installation">Installation</h2>

<h2 id="training">Training</h2>

<h2 id="eval">Evaluation</h2>

<h2 id="citing">Citing</h2>

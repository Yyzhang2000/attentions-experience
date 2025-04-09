

# About this Repository 

This is repository is an experiement on the different implementation and performance of the attention mechasim. The code is most for the learning purpose. 

## Basic Setup 

This the base set for the model used to evaluate the different attention mechanism. Noted that due to the resource constraints, some attention designed for the long context might not perform good. 

---
Model Structure, the model is like the GPT-Style model, which is the deocder with causal attention. 
`n_layers`: 
`n_heads`: 
`d_model`: 

And trained use the Shakespeare dataset 



## Multi Headed Attention
- Implemented in the `mha.py` file. 
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)


## Multi Query Attention / Group Query Attention
- Implemented in the `mqa.py` file 
- Paper: 
  - MQA: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
  - GQA: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)


## Flash Attention
- Implemented in the `fa` directory, use Pytorch built in Flash Attention 
- Paper: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)


## Multi Headed Latent  Attention
- Implemented in the `mla.py`
- Paper: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)



## Linear Attention
- Implemented in the `la.py`
- Paper: [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)


## Sliding Window Attention
- Implemented in the `swa.py`
- Paper: [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)


## Locality Sensitive Hashing Attention
- Implemented in the `lsha.py`
- Paper: [REFORMER: THE EFFICIENT TRANSFORMER](https://arxiv.org/pdf/2001.04451)



## Multi-Token Attention
- Implemented in the `mta.py`
- Paper: [Multi-Token Attention](https://arxiv.org/abs/2504.00927)


# Models

## llama

LlamaModel:  

1. Embedding

2. LlamaRotaryEmbedding (init:inv_freq, attention_scaling; forward: cos, sin)

3. ModulesList:  
   - LlamaDecoderLayers:  
     - LlamaRMS (input)
     - LlamaAttention:  
       - $W_q, W_k, W_v, W_o$
       - LlamaRotaryEmbedding (if postion_emb is None, use it)
     - LlamaRMS (post_attention)
     - LlamaMLP (SwiGLU: $W_{down}(act(W_{gate}X)*W_{up}X)$)

4. LlamaRMS (output)

5. return BaseModelOutputWithPast (output)
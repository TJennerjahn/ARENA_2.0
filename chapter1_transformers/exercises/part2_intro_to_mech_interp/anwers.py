#%%
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("mps" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


#%%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

#%%
print(gpt2_small.cfg)

#%%
model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

logits, loss = gpt2_small(model_description_text, return_type="both")
print("Model logits:", logits)
print("Logits shape", logits.shape)
print("Model loss:", loss)

#%%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))

#%%
gpt2_small.forward("Hello World")

#%%
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
# YOUR CODE HERE - get the model's prediction on the text
true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
num_correct = (prediction == true_tokens).sum()

print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
print(f"Correct words: {gpt2_small.to_str_tokens(prediction[prediction == true_tokens])}")

#%%
print(gpt2_small.to_str_tokens("HookedTransformer", prepend_bos=False))

#%%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

for key in gpt2_cache.keys():
    print(key)

#%%
layer0_pattern_from_cache = gpt2_cache["pattern", 0]
q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
seq, nhead, headsize = q.shape
layer0_attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
mask = t.triu(t.ones((seq, seq), dtype=bool), diagonal=1).to(device)
layer0_attn_scores.masked_fill_(mask, -1e9)
layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)

# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

#%%
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=gpt2_str_tokens, 
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(12)],
))

#%%
neuron_activations_for_all_layers = t.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens,
    activations=neuron_activations_for_all_layers
)

#%%
neuron_activations_for_all_layers_rearranged = utils.to_numpy(einops.rearrange(neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"))

cv.topk_tokens.topk_tokens(
    # Some weird indexing required here ¯\_(ツ)_/¯
    tokens=[gpt2_str_tokens], 
    activations=neuron_activations_for_all_layers_rearranged,
    max_k=7, 
    first_dimension_name="Layer", 
    third_dimension_name="Neuron",
    first_dimension_labels=list(range(12))
)

#%%
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)

#%%
weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

if not weights_dir.exists():
    url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
    output = str(weights_dir)
    gdown.download(url, output)

#%%
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_dir, map_location=device)
model.load_state_dict(pretrained_weights)

#%%
# text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
# text = "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense.

text = "Mr. Dursley was the director of a ﬁrm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no ﬁner boy anywhere."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

#%%
str_tokens = model.to_str_tokens(text)

def vis_heads(model, cache, sequence):
    if isinstance(sequence, str):
        sequence = model.to_str_tokens(sequence)
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]

        display(cv.attention.attention_patterns(
            tokens=str_tokens, 
            attention=attention_pattern,
        ))

# Patterns that show up:
# - Resting Pattern. All tokens attend to the BOS token
# - ID Pattern. All tokens attend to themselves
# - Previous token. Tokens attend to their preceding tokens

#%%

# cv.attention.attention_pattern(
#     tokens=str_tokens,
#     attention=attention_pattern[0]
# )

t.set_printoptions(threshold=10_000)
attention_pattern = cache["pattern", 1]
print(attention_pattern[6])
print("attention_pattern.shape", attention_pattern.shape)

#%%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    result = []
    for layer in range(model.cfg.n_layers):
        attention_patterns = cache["pattern", layer]
        for idx, pattern in enumerate(attention_patterns):
            id = t.eye(pattern.shape[-1])

            diag = pattern.diagonal()
            avg_attn = diag.mean()
            # print(f"{layer}.{idx} avg_attn = {avg_attn}")
            if avg_attn > 0.4:
                result.append(f"{layer}.{idx}")

            # frobenius_norm = t.linalg.norm(pattern - t.eye(pattern.shape[-1], dtype=float), 'fro')
            # print(f"{layer}.{idx} avg_attn = {frobenius_norm}")
            # if frobenius_norm < 9:
            #     result.append(f"{layer}.{idx}")
    return result

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    result = []
    for layer in range(model.cfg.n_layers):
        attention_patterns = cache["pattern", layer]
        for idx, pattern in enumerate(attention_patterns):
            id = t.eye(pattern.shape[-1])

            diag = pattern.diagonal(offset=-1)
            avg_attn = diag.mean()
            # print(f"{layer}.{idx} avg_attn = {avg_attn}")
            if avg_attn > 0.6:
                result.append(f"{layer}.{idx}")

    return result

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    result = []
    for layer in range(model.cfg.n_layers):
        attention_patterns = cache["pattern", layer]
        for idx, pattern in enumerate(attention_patterns):
            id = t.eye(pattern.shape[-1])

            first_col = pattern[:, 0]
            avg_attn = first_col.mean()
            if avg_attn > 0.6:
                result.append(f"{layer}.{idx}")

            # print(f"{layer}.{idx} avg_attn = {avg_attn}")
    return result


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

#%%
import random

#%%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    sequences = t.randint(1, model.cfg.d_vocab, size=(batch, seq_len), dtype=t.long)
    sequences = einops.repeat(sequences, 'batch seq -> batch (3 seq)')
    sequences = t.cat((prefix, sequences), dim=1)
    # print(sequences.shape)
    return sequences

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    sequence = generate_repeated_tokens(model, seq_len=seq_len, batch=batch)
    logits, cache = model.run_with_cache(sequence, remove_batch_dim=False)
    # print("logits.shape", logits.shape)
    return(sequence, logits, cache)



seq_len = 50
batch = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

#%%

# vis_heads(model, cache, rep_str)
for layer in range(model.cfg.n_layers):
    attention_pattern = rep_cache["pattern", layer]

    display(cv.attention.attention_patterns(
        tokens=rep_str, 
        attention=attention_pattern,
    ))

#%%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    result = []
    for layer in range(model.cfg.n_layers):
        attention_patterns = cache["pattern", layer]
        for idx, pattern in enumerate(attention_patterns):
            id = t.eye(pattern.shape[-1])

            seq_len = (attention_pattern.shape[-1] - 1) // 2
            diag = pattern.diagonal(offset=-(seq_len-1))
            avg_attn = diag.mean()
            # print(f"{layer}.{idx} avg_attn = {avg_attn}")
            if avg_attn > 0.6:
                result.append(f"{layer}.{idx}")

    return result


print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

#%%
seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
# induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    seq_len = (pattern.shape[-1] -1) // 2
    diag = pattern.diagonal(offset=1 - seq_len, dim1=-2, dim2=-1)
    # print("diag.pattern", diag.shape)
    # avg_attn = diag.mean(dim=-1).mean(dim=0)
    avg_attn = einops.reduce(diag, 'batch n_heads dest_pos -> n_heads', reduction='mean')
    induction_score_store[hook.layer(), :] = avg_attn
    # print("avg_attn.shape", avg_attn.shape)
    # print("pattern.shape", pattern.shape)
    # print("id.shape", id.shape)
    # print("induction_score.shape", induction_score_store.shape)


pattern_hook_names_filter = lambda name: name.endswith("pattern")

#%%
# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)

#%%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )


seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(gpt2_small, seq_len, batch)
# print("rep_tokens_10.shape", rep_tokens_10.shape)

# YOUR CODE HERE - find induction heads in gpt2_small
gpt2_small.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=900
)

#%%
# print(t.where(induction_score_store > .8))
# for layer, head in t.where(induction_score_store > 0.8):
for induction_head_layer in [5, 6, 7]:
    gpt2_small.run_with_hooks(
        rep_tokens_10,
        return_type=None,
        fwd_hooks=[
            (utils.get_act_name("pattern", induction_head_layer), visualize_pattern_hook)
        ]
    )

#%%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]
    # SOLUTION
    direct_attributions = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")
    l1_attributions = einops.einsum(W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead")
    l2_attributions = einops.einsum(W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead")
    return t.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1)


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")

#%%
embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

plot_logit_attribution(model, logit_attr, tokens)

#%%
from timeit import timeit
#%%
seq_len = 50

embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]
first_half_tokens = rep_tokens[0, : 1 + seq_len]
second_half_tokens = rep_tokens[0, seq_len:]

tokens = t.concat([first_half_tokens, second_half_tokens[1:]])

# YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
# first_half_logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens)[:50]
# second_half_logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens)[50:]
# print("first_half_logit_attr.shape", first_half_logit_attr.shape)
# print("second_half_logit_attr.shape", second_half_logit_attr.shape)
first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)
assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")


# full_attribution = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens)
# plot_logit_attribution(model, full_attribution, tokens, "Logit attribution (full)")

#%%
def head_ablation_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_head"]:
    print("head_index", head_index_to_ablate)
    v[:,:,head_index_to_ablate,:] = 0
    return v


def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("v", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)

imshow(
    ablation_scores, 
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Logit Difference After Ablating Heads", 
    text_auto=".2f",
    width=900, height=400
)

#%%
A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")

#%%
print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)
print()
print("Singular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)
print("Full SVD:")
print(AB_factor.svd())

#%%
C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C
print("Unfactored:", ABC.shape, ABC.norm())
print("Factored:", ABC_factor.shape, ABC_factor.norm())
print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")

#%%
AB_unfactored = AB_factor.AB
t.testing.assert_close(AB_unfactored, AB)

#%%
layer = 1
head_index = 4

W_O = model.W_O[layer, head_index]
W_V = model.W_V[layer, head_index]
W_E = model.W_E
W_U = model.W_U

OV_circuit = FactoredMatrix(W_V, W_O)
full_OV_circuit = W_E @ OV_circuit @ W_U

tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

#%%
print(full_OV_circuit.shape)

#%%
# YOUR CODE HERE - get a random sample from the full OV circuit, so it can be plotted with `imshow`
random_indices = t.randperm(full_OV_circuit.shape[0])[:200]
full_OV_circuit_sample = full_OV_circuit[random_indices, random_indices].AB
imshow(
    full_OV_circuit_sample,
    labels={"x": "Input token", "y": "Logits on output token"},
    title="Full OV circuit for copying head",
    width=700,
    height=700,
)

#%%
from timeit import timeit

#%%
def top_1_acc(full_OV_circuit: FactoredMatrix) -> float:
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    count = 0
    for row in range(full_OV_circuit.A.shape[0]):
        row_max = t.argmax(full_OV_circuit.A[row, :] @ full_OV_circuit.B)
        if Tensor([row]) == row_max:
            count += 1
    accuracy = count / full_OV_circuit.A.shape[0]
    return accuracy



print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")

#%%
def top_1_acc_iteration(full_OV_circuit: FactoredMatrix, batch_size: int = 100) -> float: 
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    # SOLUTION
    A, B = full_OV_circuit.A, full_OV_circuit.B
    nrows = full_OV_circuit.shape[0]
    nrows_max_on_diagonal = 0

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        rng = range(i, min(i + batch_size, nrows))
        if rng:
            submatrix = A[rng, :] @ B
            diag_indices = t.tensor(rng, device=submatrix.device)
            nrows_max_on_diagonal += (submatrix.argmax(-1) == diag_indices).float().sum().item()

    return nrows_max_on_diagonal / nrows


print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc_iteration(full_OV_circuit):.4f}")

#%%
W_O_both = einops.rearrange(model.W_O[1, [4,10]], 'head d_head d_model -> (head d_head) d_model')
W_V_both = einops.rearrange(model.W_V[1, [4,10]], 'head d_model d_head -> d_model (head d_head)')

W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U

print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc_iteration(W_OV_eff):.4f}")

#%%
def mask_scores(attn_scores: Float[Tensor, "query_nctx key_nctx"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    assert attn_scores.shape == (model.cfg.n_ctx, model.cfg.n_ctx)
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores



# YOUR CODE HERE - calculate the matrix `pos_by_pos_pattern` as described above
full_QK_circuit = model.W_pos @ model.W_Q[0, 7] @ (model.W_K[0,7].T) @ model.W_pos.T

pos_by_pos_pattern = t.softmax(mask_scores(full_QK_circuit / model.cfg.d_head**0.5), dim=1)

tests.test_pos_by_pos_pattern(pos_by_pos_pattern, model, layer, head_index)

#%%
print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")

imshow(
    utils.to_numpy(pos_by_pos_pattern[:100, :100]), 
    labels={"x": "Key", "y": "Query"}, 
    title="Attention patterns for prev-token QK circuit, first 100 indices",
    width=700,
    height=700
)

#%%
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, :, :]th element is y_i (from notation above)
    '''
    y0 = cache["embed"].unsqueeze(0)
    y1 = cache["pos_embed"].unsqueeze(0)
    y_rest = cache["result", 0].transpose(0, 1)

    return t.cat([y0, y1, y_rest], dim=0)

def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    W_Q = model.W_Q[1, ind_head_index]

    return einops.einsum(
        decomposed_qk_input, W_Q,
        "n seq d_model, d_model d_head -> n seq d_head"
    )

def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_K (so the sum along axis 0 is just the k-values)
    '''
    W_K = model.W_K[1, ind_head_index]

    return einops.einsum(
        decomposed_qk_input, W_K,
        "n seq d_model, d_model d_head -> n seq d_head"
    )


ind_head_index = 4
# First we get decomposed q and k input, and check they're what we expect
decomposed_qk_input = decompose_qk_input(rep_cache)
decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
# Second, we plot our results
component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
    imshow(
        utils.to_numpy(decomposed_input.pow(2).sum([-1])), 
        labels={"x": "Position", "y": "Component"},
        title=f"Norms of components of {name}", 
        y=component_labels,
        width=1000, height=400
    )

#%%
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, :, :]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    '''
    return einops.einsum(
        decomposed_q, decomposed_k,
        "q_comp q_pos d_model, k_comp k_pos d_model -> q_comp k_comp q_pos k_pos",
    )


tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k)


#%%
decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
decomposed_stds = einops.reduce(
    decomposed_scores, 
    "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", 
    t.std
)

# First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
imshow(
    utils.to_numpy(t.tril(decomposed_scores[0, 9])), 
    title="Attention score contributions from (query, key) = (embed, output of L0H7)",
    width=800,
    height=800,
)

# Second plot: std dev over query and key positions, shown by component
imshow(
    utils.to_numpy(decomposed_stds), 
    labels={"x": "Key Component", "y": "Query Component"},
    title="Standard deviations of attention score contributions (by key and query component)", 
    x=component_labels, 
    y=component_labels,
    width=800,
    height=800,
)

#%%
def find_K_comp_full_circuit(
    model: HookedTransformer,
    prev_token_head_index: int,
    ind_head_index: int
) -> FactoredMatrix:
    '''
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    # SOLUTION
    W_E = model.W_E
    W_Q = model.W_Q[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]

    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K
    return FactoredMatrix(Q, K.T)


prev_token_head_index = 7
ind_head_index = 10
K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc_iteration(K_comp_circuit.T):.4f}")
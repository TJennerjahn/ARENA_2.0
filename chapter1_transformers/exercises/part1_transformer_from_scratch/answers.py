#%%
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import webbrowser

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

# device = t.device("cuda" if t.cuda.is_available() else "cpu")
device = "cpu"

MAIN = __name__ == '__main__'


if MAIN:
	reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

# %% 1️⃣ UNDERSTANDING INPUTS & OUTPUTS OF A TRANSFORMER


if MAIN:
	sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
	print(sorted_vocab[:20])
	print()
	print(sorted_vocab[250:270])
	print()
	print(sorted_vocab[990:1010])
	print()

# %%


if MAIN:
	print(sorted_vocab[-20:])

# %%


if MAIN:
	print(reference_gpt2.to_str_tokens("Ralph"))
	print(reference_gpt2.to_str_tokens(" Ralph"))
	print(reference_gpt2.to_str_tokens(" ralph"))
	print(reference_gpt2.to_str_tokens("ralph"))

# %%


if MAIN:
	print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))

# %%


if MAIN:
	reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
	tokens = reference_gpt2.to_tokens(reference_text).to(device)


	print(tokens)
	print("tokens.shape: ", tokens.shape)
	print(reference_gpt2.to_str_tokens(tokens))

# %%


if MAIN:
	logits, cache = reference_gpt2.run_with_cache(tokens)
	print(logits.shape)

# %%


if MAIN:
	probs = logits.softmax(dim=-1)
	print(probs.shape)

# %%


if MAIN:
	most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
	
	print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

# %%


if MAIN:
	next_token = logits[0, -1].argmax(dim=-1)
	next_char = reference_gpt2.to_string(next_token)
	print(repr(next_char))

# %%


if MAIN:
	print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")
	
	for i in range(10):
		print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
		# Define new input sequence, by appending the previously generated token
		tokens = t.cat([tokens, next_token[None, None]], dim=-1)
		# Pass our new sequence through the model, to get new output
		logits = reference_gpt2(tokens)
		# Get the predicted token at the end of our sequence
		next_token = logits[0, -1].argmax(dim=-1)
		# Decode and print the result
		next_char = reference_gpt2.to_string(next_token)

# %% 2️⃣ CLEAN TRANSFORMER IMPLEMENTATION


if MAIN:
	for activation_name, activation in cache.items():
		# Only print for first layer
		if ".0." in activation_name or "blocks" not in activation_name:
			print(f"{activation_name:30} {tuple(activation.shape)}")

# %%


if MAIN:
	for name, param in reference_gpt2.named_parameters():
		# Only print for first layer
		if ".0." in name or "blocks" not in name:
			print(f"{name:18} {tuple(param.shape)}")

# %%

# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures

if MAIN:
	print(reference_gpt2.cfg)

# %%

@dataclass
class Config:
	d_model: int = 768
	debug: bool = True
	layer_norm_eps: float = 1e-5
	d_vocab: int = 50257
	init_range: float = 0.02
	n_ctx: int = 1024
	d_head: int = 64
	d_mlp: int = 3072
	n_heads: int = 12
	n_layers: int = 12


if MAIN:
	cfg = Config()
	print(cfg)

# %%

def rand_float_test(cls, shape):
	cfg = Config(debug=True)
	layer = cls(cfg).to(device)
	random_input = t.randn(shape).to(device)
	print("Input shape:", random_input.shape)
	output = layer(random_input)
	if isinstance(output, tuple): output = output[0]
	print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
	cfg = Config(debug=True)
	layer = cls(cfg).to(device)
	random_input = t.randint(100, 1000, shape).to(device)
	print("Input shape:", random_input.shape)
	output = layer(random_input)
	if isinstance(output, tuple): output = output[0]
	print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
	cfg = Config(debug=True)
	layer = cls(cfg).to(device)
	layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
	print("Input shape:", input.shape)
	output = layer(input)
	if isinstance(output, tuple): output = output[0]
	print("Output shape:", output.shape)
	try: reference_output = gpt2_layer(input)
	except: reference_output = gpt2_layer(input, input, input)
	print("Reference output shape:", reference_output.shape, "\n")
	comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
	print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")

# %%

class LayerNorm(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.w = nn.Parameter(t.ones(cfg.d_model))
		self.b = nn.Parameter(t.zeros(cfg.d_model))

	def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
		residual_mean = residual.mean(dim=-1, keepdim=True)
		residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

		residual = (residual - residual_mean) / residual_std
		return residual * self.w + self.b



if MAIN:
	rand_float_test(LayerNorm, [2, 4, 768])
	load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])

# %%

class Embed(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
		nn.init.normal_(self.W_E, std=self.cfg.init_range)

	def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
		# print("tokens.shape: ", tokens.shape)
		# print("self.W_E.shape: ", self.W_E.shape)
		# print("tokens: ", tokens)
		# print("tokens as string", [reference_gpt2.to_str_tokens(token_entry) for token_entry in tokens])

		embedding_vector = self.W_E[tokens[0][0]]
		# print("embedding_vector: ", embedding_vector)
		# print("result", self.W_E[tokens[0][0]])


		return self.W_E[tokens]



if MAIN:
	rand_int_test(Embed, [2, 4])
	load_gpt2_test(Embed, reference_gpt2.embed, tokens)

# %%

class PosEmbed(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
		nn.init.normal_(self.W_pos, std=self.cfg.init_range)

	def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
		batch, seq_len = tokens.shape
		return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)


if MAIN:
	rand_int_test(PosEmbed, [2, 4])
	load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

# %%

import circuitsvis as cv
from IPython.display import display


if MAIN:
	html = cv.attention.attention_patterns(
		tokens=reference_gpt2.to_str_tokens(reference_text), 
		attention=cache["pattern", 0][0]
	)
	display(html)

# %%

class Attention(nn.Module):
	IGNORE: Float[Tensor, ""]

	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
		self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
		self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
		self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
		self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
		nn.init.normal_(self.W_Q, std=self.cfg.init_range)
		nn.init.normal_(self.W_K, std=self.cfg.init_range)
		nn.init.normal_(self.W_V, std=self.cfg.init_range)
		nn.init.normal_(self.W_O, std=self.cfg.init_range)
		self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

	def forward(
		self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
	) -> Float[Tensor, "batch posn d_model"]:
		# Calculate query, key and value vectors
		q = einops.einsum(
			normalized_resid_pre, self.W_Q,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_Q
		k = einops.einsum(
			normalized_resid_pre, self.W_K,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_K
		v = einops.einsum(
			normalized_resid_pre, self.W_V,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_V

		# Calculate attention scores, then scale and mask, and apply softmax to get probabilities
		attn_scores = einops.einsum(
			q, k,
			"batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K", 
		)
		attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
		attn_pattern = attn_scores_masked.softmax(-1)

		# Take weighted sum of value vectors, according to attention probabilities
		z = einops.einsum(
			v, attn_pattern,
			"batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head", 
		)

		# Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
		attn_out = einops.einsum(
			z, self.W_O,
			"batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model", 
		) + self.b_O

		return attn_out

	def apply_causal_mask(
		self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
	) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
		'''
		Applies a causal mask to attention scores, and returns masked scores.
		'''
		# Define a mask that is True for all positions we want to set probabilities to zero for
		all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
		mask = t.triu(all_ones, diagonal=1).bool()
		# Apply the mask to attention scores, then return the masked scores
		attn_scores.masked_fill_(mask, self.IGNORE)
		return attn_scores



if MAIN:
	rand_float_test(Attention, [2, 4, 768])
	load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

# %%

class MLP(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
		self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
		self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
		self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
		nn.init.normal_(self.W_in, std=self.cfg.init_range)
		nn.init.normal_(self.W_out, std=self.cfg.init_range)

	def forward(
		self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
	) -> Float[Tensor, "batch posn d_model"]:
		pre = einops.einsum(
			normalized_resid_mid, self.W_in,
			"batch position d_model, d_model d_mlp -> batch position d_mlp", 
		) + self.b_in
		post = gelu_new(pre)
		mlp_out = einops.einsum(
			post, self.W_out,
			"batch position d_mlp, d_mlp d_model -> batch position d_model", 
		) + self.b_out
		return mlp_out



if MAIN:
	rand_float_test(MLP, [2, 4, 768])
	load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])

# %%

class TransformerBlock(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.ln1 = LayerNorm(cfg)
		self.attn = Attention(cfg)
		self.ln2 = LayerNorm(cfg)
		self.mlp = MLP(cfg)

	def forward(
		self, resid_pre: Float[Tensor, "batch position d_model"]
	) -> Float[Tensor, "batch position d_model"]:
		resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
		resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
		return resid_post
		
		

if MAIN:
	rand_float_test(TransformerBlock, [2, 4, 768])
	load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

# %%

class Unembed(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
		nn.init.normal_(self.W_U, std=self.cfg.init_range)
		self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

	def forward(
		self, normalized_resid_final: Float[Tensor, "batch position d_model"]
	) -> Float[Tensor, "batch position d_vocab"]:
		return einops.einsum(
			normalized_resid_final, self.W_U,
			"batch posn d_model, d_model d_vocab -> batch posn d_vocab",
		) + self.b_U
		# Or, could just do `normalized_resid_final @ self.W_U + self.b_U`



if MAIN:
	rand_float_test(Unembed, [2, 4, 768])
	load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

# %%

class DemoTransformer(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.embed = Embed(cfg)
		self.pos_embed = PosEmbed(cfg)
		self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
		self.ln_final = LayerNorm(cfg)
		self.unembed = Unembed(cfg)

	def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
		residual = self.embed(tokens) + self.pos_embed(tokens)
		for block in self.blocks:
			residual = block(residual)
		logits = self.unembed(self.ln_final(residual))
		return logits



if MAIN:
	rand_int_test(DemoTransformer, [2, 4])
	load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

# %%


if MAIN:
	demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
	demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
	
	demo_logits = demo_gpt2(tokens)

# %%

def get_log_probs(
	logits: Float[Tensor, "batch posn d_vocab"], 
	tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
	
	log_probs = logits.log_softmax(dim=-1)
	# Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
	log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

	return log_probs_for_tokens



if MAIN:
	pred_log_probs = get_log_probs(demo_logits, tokens)
	print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
	print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
	print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# %%


if MAIN:
	test_string = '''The Total Perspective Vortex derives its picture of the whole Universe on the principle of'''
	for i in tqdm(range(100)):
		test_tokens = reference_gpt2.to_tokens(test_string).to(device)
		demo_logits = demo_gpt2(test_tokens)
		test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
	
	print(test_string)

# %% 3️⃣ TRAINING A TRANSFORMER


if MAIN:
	model_cfg = Config(
		debug=False, 
		d_model=256, 
		n_heads=4, 
		d_head=64, 
		d_mlp=1024, 
		n_layers=2, 
		n_ctx=256, 
		d_vocab=reference_gpt2.cfg.d_vocab
	)
	model = DemoTransformer(model_cfg)

# %%

@dataclass
class TransformerTrainingArgs():
	batch_size = 32
	epochs = 5
	max_steps_per_epoch = 500
	lr = 1e-3
	weight_decay = 1e-2
	wandb_project: Optional[str] = "day1-demotransformer"
	wandb_name: Optional[str] = None


if MAIN:
	args = TransformerTrainingArgs()

# %%


if MAIN:
	dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
	print(dataset)
	print(dataset[0]['text'][:100])


# %%

if MAIN:
	tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model.cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
	
	dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
	train_loader = DataLoader(dataset_dict["train"], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = DataLoader(dataset_dict["test"], batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# %%


if MAIN:
	first_batch = train_loader.dataset[:args.batch_size]
	
	print(first_batch.keys())
	print(first_batch['tokens'].shape)

# %%

class TransformerTrainer:
	def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
		super().__init__()
		self.model = model
		self.args = args
		self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		self.step = 0


	def training_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
		'''
		Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

		Remember that `batch` is a dictionary with the single key 'tokens'.
		'''
		tokens = batch["tokens"].to(device)
		logits = self.model(tokens)
		loss = -get_log_probs(logits, tokens).mean()
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		self.step += 1
		wandb.log({"train_loss": loss}, step=self.step)
		return loss


	def validation_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]):
		'''
		Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
		is correct). Logging should happen in the `train` function (after we've computed the accuracy for 
		the whole validation set).
		'''
		tokens = batch["tokens"].to(device)
		logits: Tensor = self.model(tokens)[:, :-1]
		predicted_tokens = logits.argmax(dim=-1)
		correct_predictions = (predicted_tokens == tokens[:, 1:]).flatten()
		return correct_predictions
	
	def train(self):
		'''
		Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
		for each epoch at `self.args.max_steps_per_epoch` steps.
		'''
		wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
		accuracy = np.nan

		progress_bar = tqdm(total = self.args.max_steps_per_epoch * self.args.epochs)

		for epoch in range(self.args.epochs):
			for i, batch in enumerate(self.train_loader()):
				loss = self.training_step(batch)
				progress_bar.update()
				progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")
				if i >= self.args.max_steps_per_epoch:
					break

			correct_predictions = t.concat([self.validation_step(batch) for batch in self.test_loader()])
			accuracy = correct_predictions.float().mean().item()
			wandb.log({"accuracy": accuracy}, step=self.step)
			
		wandb.finish()


	def train_loader(self) -> DataLoader:
		'''Returns train loader (as in code above).'''
		return DataLoader(dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


	def test_loader(self) -> DataLoader:
		'''Returns test loader (as in code above).'''
		return DataLoader(dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)



# %%

if MAIN:
	model = DemoTransformer(model_cfg).to(device)
	args = TransformerTrainingArgs()
	trainer = TransformerTrainer(args, model)
	trainer.train()

#%%
# plot different torch activiation functions
import torch as t
import plotly.graph_objects as go
import plotly.express as px
import torch.nn.functional as F

x = t.linspace(-5, 5, 100)
y_1 = F.relu(x)
y_2 = F.leaky_relu(x)
y_3 = F.elu(x)
y_4 = F.gelu(x)
y_5 = F.silu(x)
y_6 = F.softplus(x)
y_7 = F.sigmoid(x)

# label the lineplots like the function names
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_1, name="relu"))
fig.add_trace(go.Scatter(x=x, y=y_2, name="leaky_relu"))
fig.add_trace(go.Scatter(x=x, y=y_3, name="elu"))
fig.add_trace(go.Scatter(x=x, y=y_4, name="gelu"))
fig.add_trace(go.Scatter(x=x, y=y_5, name="silu"))
fig.add_trace(go.Scatter(x=x, y=y_6, name="softplus"))
fig.add_trace(go.Scatter(x=x, y=y_7, name="sigmoid"))

fig.update_layout(
	title="Activation functions",
	xaxis_title="x",
	yaxis_title="y",
	legend_title="Function",
)

fig.show()

#%%
import numpy as np
x = np.linspace(-10, 10, 100000)
y = np.log(x)

fig = px.line(x=x, y=y, title="Log")
fig.show()

#%%
from collections import defaultdict, Counter
import re

def get_stats(vocab):
	"""Count the frequency of token pairs in the vocabulary."""
	pairs = defaultdict(int)
	for word, freq in vocab.items():
		symbols = word.split()
		for i in range(len(symbols) - 1):
			pairs[symbols[i], symbols[i + 1]] += freq
	return pairs

def merge_vocab(pair, vocab):
	"""Merge the most frequent token pair in the vocabulary."""
	first, second = pair
	new_token = ''.join(pair)
	new_vocab = {}
	pattern = re.escape(first + ' ' + second)
	replacement = new_token
	for word in vocab:
		new_word = re.sub(pattern, replacement, word)
		new_vocab[new_word] = vocab[word]
	return new_vocab

def get_vocab(text):
	"""Initialize the vocabulary with individual words from the text."""
	# Add spaces around each word and end with </w> to indicate word boundaries
	words = text.split()
	vocab = {' '.join(word) + ' </w>': 1 for word in words}
	return vocab

def bpe_tokenizer(text, num_merges=10):
	"""Perform Byte-Pair Encoding tokenization."""
	vocab = get_vocab(text)
	for i in range(num_merges):
		pairs = get_stats(vocab)
		if not pairs:
			break
		best_pair = max(pairs, key=pairs.get)
		vocab = merge_vocab(best_pair, vocab)
	# Extract the tokens from the vocabulary
	tokens = set()
	for word in vocab:
		tokens.update(word.split())
	return tokens

# Sample text for BPE tokenization
text = """
Machine learning is a method of data analysis that automates analytical model building. 
It is a branch of artificial intelligence based on the idea that systems can learn from data, 
identify patterns and make decisions with minimal human intervention.
"""

# Tokenize using BPE until the vocabulary size is 50 or less
bpe_tokens = bpe_tokenizer(text, num_merges=200)
final_tokens = list(bpe_tokens)

print("Vocab Size:", len(final_tokens))
print(final_tokens)

#%%
model_cfg = Config()
model = DemoTransformer(model_cfg).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)

tokenizer = reference_gpt2.tokenizer

class TransformerSampler:

	def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
		self.model = model
		self.cfg = model.cfg
		self.tokenizer = tokenizer

	@t.inference_mode()
	def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs):
		'''
		Returns a string of autoregressively generated text, starting from the prompt.

		Sampling terminates at max_tokens_generated, or when the model generates an
		end-of-sequence token.

		kwargs are passed to sample_next_token, to give detailed instructions on how 
		new tokens are chosen.
		'''
		model.eval()
		tokenized_prompt = self.tokenizer.encode(prompt)
		tokens_generated = 0


		# print("tokenized_prompt: ", tokenized_prompt)
		# print("eos_token_id: ", self.tokenizer.eos_token_id)
		# print("last token: ", tokenized_prompt[-1])

		while tokens_generated < max_tokens_generated and tokenized_prompt[-1] != self.tokenizer.eos_token_id:
			# Pass the tokenized prompt through the model to get logits
			model_input = t.tensor(tokenized_prompt).unsqueeze(0).to(device)
			print("model_input.shape: ", model_input.shape)

			logits = self.model(model_input)
			print("logits.shape: ", logits.shape)

			# Take the logit vector for the last token in the prompt
			logit_vector = logits[0, -1]
			print("logit_vector.shape: ", logit_vector.shape)

			# Sampling from this distribution to get a new token
			next_token = self.sample_next_token(model_input.squeeze(), logit_vector, **kwargs)
			print("next_token: ", next_token)

			# append the new token to the prompt
			tokenized_prompt.append(next_token)
			tokens_generated += 1
		
		# Decode the tokenized prompt into a string
		return self.tokenizer.decode(tokenized_prompt)


	@t.inference_mode()
	def beam_search(
		self,
		prompt: str, 
		num_return_sequences: int, 
		num_beams: int, 
		max_new_tokens: int, 
		no_repeat_ngram_size: int = 0,
		verbose=False
	) -> List[Tuple[float, t.Tensor]]:
		'''
		Returns a string of autoregressively generated text, starting from the prompt.

		Sampling terminates at max_tokens_generated, or when the model generates an
		end-of-sequence token.

		kwargs are passed to sample_next_token, to give detailed instructions on how 
		new tokens are chosen.
		'''
		pass



	@staticmethod
	def sample_next_token(
		input_ids: Int[Tensor, "seq_len"], 
		logits: Float[Tensor, "seq_len d_vocab"], 
		temperature=1.0, 
		top_k=0, 
		top_p=0.0, 
		frequency_penalty=0.0,
		seed=None
	):
		assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
		assert temperature >= 0, "Temperature should be non-negative"
		assert 0 <= top_p <= 1.0, "Top-p must be a probability"
		assert 0 <= top_k, "Top-k must be non-negative"
		assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

		# Set random seeds for reproducibility
		if seed is not None:
			t.manual_seed(seed)
			np.random.seed(seed)

		# Apply all the specialized sampling methods
		if temperature == 0:
			return TransformerSampler.greedy_search(logits)
		elif temperature != 1.0:
			logits = TransformerSampler.apply_temperature(logits, temperature)
		if frequency_penalty != 0.0:
			logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
		if top_k > 0:
			return TransformerSampler.sample_top_k(logits, top_k)
		if top_p > 0.0:
			return TransformerSampler.sample_top_p(logits, top_p)
		return TransformerSampler.sample_basic(logits)


	@staticmethod
	def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
		'''
		Returns the most likely token (as an int).
		'''
		out = logits.argmax().item()
		return out


	@staticmethod
	def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
		'''
		Applies temperature scaling to the logits.
		'''
		assert temperature > 0
		return logits / temperature

	@staticmethod
	def apply_frequency_penalty(input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float) -> Float[Tensor, "d_vocab"]:
		'''
		Applies a frequency penalty to the logits.
		'''
		freq_count = t.bincount(input_ids, minlength=logits.shape[-1])
		penalty = freq_count * freq_penalty
		logits -= penalty
		return logits

	@staticmethod
	def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
		'''
		Samples from the distribution defined by the logits.
		'''
		distribution = t.distributions.categorical.Categorical(logits=logits)
		out = distribution.sample().item()
		assert isinstance(out, int), "sample_basic should return an int"
		return out

	@staticmethod
	def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
		'''
		Samples from the top k most likely tokens.
		'''
		top_k_vals, top_k_indices = t.topk(logits, k)
		idx = t.distributions.categorical.Categorical(logits=top_k_vals).sample()
		return top_k_indices[idx].item()


	@staticmethod
	def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
		'''
		Samples from the most likely tokens which make up at least p cumulative probability.
		'''
		logits_sorted, indices = logits.sort(descending=True, stable=True)
		cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
		n_keep = t.searchsorted(cumul_probs, top_p, side="right").item() + 1
		n_keep = max(n_keep, min_tokens_to_keep)
		keep_idx = indices[:n_keep]
		keep_logits = logits[keep_idx]
		sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
		return keep_idx[sample].item()
	


#%%
sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Greedy decoding with prompt: {prompt!r}\n")

output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output!r}\n")

expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Tests passed!")
# %%

#%%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
	" church": 0.0648,
	" house": 0.0367,
	" temple": 0.0145,
	" same": 0.0104,
	" Church": 0.0097
}
frequency_of_top_5 = defaultdict(int)

N = 10_000
for _ in tqdm(range(N)):
	token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits)
	frequency_of_top_5[tokenizer.decode(token)] += 1

for word in expected_top_5:
	expected_freq = expected_top_5[word]
	observed_freq = frequency_of_top_5[word] / N
	print(f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}")
	assert abs(observed_freq - expected_freq) < 0.01, "Try increasing N if this fails by a small amount."

print("Tests passed!")

#%%
logits = t.tensor([1, 2]).log()

cold_logits = TransformerSampler.apply_temperature(logits, temperature=0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)

hot_logits = TransformerSampler.apply_temperature(logits, temperature=1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)

print("Tests passed!")

#%%
bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt")
logits = t.ones(tokenizer.vocab_size)
penalized_logits = TransformerSampler.apply_frequency_penalty(input_ids.squeeze(), logits, 2.0)

assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space, 1-2*6=-11"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space, 1-2*3=-5"

print("Tests passed!")

#%%
sampler = TransformerSampler(model, tokenizer)

N_RUNS = 1
# your_prompt = "Jingle bells, jingle bells, jingle all the way"
your_prompt = "Once upon a time on an alien planet"
cases = [
	("High freq penalty", dict(frequency_penalty=100.0)),
	("Negative freq penalty", dict(frequency_penalty=-3.0)),
	("Too hot!", dict(temperature=2.0)),
	("Pleasantly cool", dict(temperature=0.7)),
	("Pleasantly warm", dict(temperature=0.9)),
	("Too cold!", dict(temperature=0.01)),
]

table = Table("Name", "Kwargs", "Output", title="Sampling - Manual Testing")

for (name, kwargs) in cases:
	for i in range(N_RUNS):
		output = sampler.sample(your_prompt, max_tokens_generated=100, **kwargs)
		table.add_row(name, repr(kwargs), repr(output) + "\n")

rprint(table)


#%%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
	" church": 0.0648,
	" house": 0.0367,
	" temple": 0.0145,
	" same": 0.0104,
	" Church": 0.0097
}
topk_5_sum = sum(expected_top_5.values())

observed_freqs = defaultdict(int)

N = 10000
for _ in tqdm(range(N)):
	token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_k=5)
	observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_5:
	expected_freq = expected_top_5[word] / topk_5_sum
	observed_freq = observed_freqs[word] / N
	print(f"Word: {word!r:<9}. Expected freq = {expected_freq:.4f}, observed freq = {observed_freq:.4f}")
	assert abs(observed_freq - expected_freq) < 0.015, "Try increasing N if this fails by a small amount."

#%%
sampler = TransformerSampler(model, tokenizer)

your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
output = sampler.sample(your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
rprint(f"Your model said:\n\n[bold dark_orange]{output}")

#%%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_10pct = {
	" church": 0.0648,
	" house": 0.0367, # These are the two most likely tokens, and add up to >10%
}
top_10pct_sum = sum(expected_top_10pct.values())

observed_freqs = defaultdict(int)

N = 10000
for _ in tqdm(range(N)):
	token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_p=0.1)
	observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_10pct:
	expected_freq = expected_top_10pct[word] / top_10pct_sum
	observed_freq = observed_freqs[word] / N
	print(f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}")
	assert abs(observed_freq - expected_freq) < 0.01, "Try increasing N if this fails by a small amount."

#%%
sampler = TransformerSampler(model, tokenizer)

your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
output = sampler.sample(your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
rprint(f"Your model said:\n\n[bold dark_orange]{output}")

#%%
@dataclass
class Beams:
    '''Class to store beams during beam search.'''
    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def new_beams(self, logprob_sums, tokens) -> "Beams":
        '''Creates a new Beams object with the same model and tokenizer.'''
        return Beams(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx) -> "Beams":
        '''Allows you to take a slice of the beams object along the batch dimension.'''
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> List[Tuple[float, str]]:
        '''Returns self as a list of logprob sums and completions (useful for getting final output).'''
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]


    def generate(self, toks_per_beam: int, no_repeat_ngram_size: Optional[int] = None) -> "Beams":
        '''
        Starting from the current set of beams (which has length `num_beams`), returns a new
        set of `num_beams * toks_per_beam`, containing the best `toks_per_beam` continuations for each
        of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with
        a repeating n-gram of this length.
        '''
        pass

    def filter(self, num_beams: int) -> Tuple["Beams", "Beams"]:
        '''
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `num_beams` which are also not terminated.

            early_terminations: Beams
                filtered version of self, containing all best `num_beams` which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        '''
        pass

    def print(self, title="Best completions", max_print_chars=80) -> None:
        '''
        Prints out a set of sequences with their corresponding logitsums.
        '''
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = text[:int(0.3 * max_print_chars)] + " ... " + text[-int(0.7 * max_print_chars):]
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str, 
    num_return_sequences: int, 
    num_beams: int, 
    max_new_tokens: int, 
    no_repeat_ngram_size: Optional[int] = None,
    verbose=False
) -> List[Tuple[float, Tensor]]:
    '''
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
    from the initial prompt) until either of the two stopping criteria are met:

        (1) we've generated `max_new_tokens` tokens, or
        (2) we've generated `num_returns_sequences` terminating sequences.

    To modularize this function, most of the actual complexity is in the Beams class,
    in the `generate` and `filter` methods.
    '''

    assert num_return_sequences <= num_beams
    self.model.eval()

    pass

#%%
def get_topk_non_repeating(
    self,
    logprobs: Float[Tensor, "batch d_vocab"], 
    no_repeat_ngram_size: Optional[int],
    k: int, 
) -> Tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
    '''
    logprobs: 
        tensor of the log-probs for the next token
    no_repeat_ngram_size:
        size of ngram to avoid repeating
    k:
        number of top logits to return, for each beam in our collection

    Returns:
        equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure
        that no returned tokens would produce an ngram of size  `no_repeat_ngram_size`
        which has already appeared in `self.tokens`.
    '''
    batch, seq_len = self.tokens.shape
    neg_inf = t.tensor(-1.0e4).to(device)

    # If completion isn't long enough for a repetition, or we have no restructions, just return topk
    if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size-1):
        # Otherwise, we need to check for ngram repetitions
        # First, get the most recent `no_repeat_ngram_size-1` tokens
        last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size-1):]
        # Next, find all the tokens we're not allowed to generate (by going iterating through past ngrams and seeing if those ngram prefixes match the last one)
        for i in range(seq_len - (no_repeat_ngram_size-1)):
            ngrams = self.tokens[:, i:i+no_repeat_ngram_size] # (batch, ngram)
            ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(-1) # (batch,)
            ngram_end_tokens = ngrams[:, [-1]] # (batch, 1)
            # Fill logprobs with neginf wherever the ngrams are repeated
            logprobs[range(batch), ngram_end_tokens] = t.where(
                ngrams_are_repeated,
                neg_inf,
                logprobs[range(batch), ngram_end_tokens],
        )

    # Finally, get our actual tokens
    return logprobs.topk(k=k, dim=-1)

#%%
@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str, 
    num_return_sequences: int, 
    num_beams: int, 
    max_new_tokens: int, 
    no_repeat_ngram_size: Optional[int] = None,
    verbose=False
) -> List[Tuple[float, Tensor]]:
    '''
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
    from the initial prompt) until either of the two stopping criteria are met:

        (1) we've generated `max_new_tokens` tokens, or
        (2) we've generated `num_returns_sequences` terminating sequences.

    To modularize this function, most of the actual complexity is in the Beams class,
    in the `generate` and `filter` methods.
    '''

    assert num_return_sequences <= num_beams
    self.model.eval()

    # SOLUTION
    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

    # List for final beams to return (and early terminations)
    final_logprobs_and_completions: List[Tuple[float, str]] = []
    # Keep track of all best beams after each step
    best_beams = Beams(self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens)

    for n in tqdm(range(max_new_tokens)):

        # Generation step
        best_beams = best_beams.generate(toks_per_beam=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)

        # Filtering step
        best_beams, best_beams_terminated = best_beams.filter(num_beams=num_beams)
        final_logprobs_and_completions.extend(best_beams_terminated.logprobs_and_completions)

        # Print output
        if verbose:
            best_beams.print()

        # Check stopping condition
        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    final_logprobs_and_completions = final_logprobs_and_completions[:num_return_sequences]
    return final_logprobs_and_completions

#%%
TransformerSampler.beam_search = beam_search

sampler = TransformerSampler(model, tokenizer)

prompt = "The ships hung in the sky in much the same way that"
orig_len = len(tokenizer.encode(prompt))

final_logitsums_and_completions = sampler.beam_search(
    prompt=prompt, 
    num_return_sequences=3,
    num_beams=40,
    max_new_tokens=60, 
    no_repeat_ngram_size=2,
    verbose=False
)

# Print all the best output
for logprob_sum, text in final_logitsums_and_completions:
    avg_logprob_as_prob = t.tensor(logprob_sum / (len(tokenizer.encode(text)) - orig_len)).exp().item()
    print("=" * 25 + f" Avg logprob (as probability) = {avg_logprob_as_prob:.3f} " + "=" * 25)
    rprint("Best output:\n\n[bold dark_orange]" + text)
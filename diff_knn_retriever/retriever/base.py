import dataclasses
import itertools
import json
import logging
import os
import random
from typing import Any, Callable, Iterable, Literal, Optional, Union

import accelerate
import datasets
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


def my_cos_sim(v1, v2):
    """
    Compute cosine similarity between two tensors

    Parameters
    ----------
    v1 : torch.Tensor
        Tensor of shape (batch_size, #vector, hidden_dim)
    v2 : torch.Tensor
        Tensor of shape (memory_size, #vector, hidden_dim)

    Returns
    -------
    cos_sim : torch.Tensor
        Tensor of shape (batch_size, memory_size, #vector)
    """

    batch_size = v1.size(0)
    memory_size = v2.size(0)
    num_vectors = v1.size(1)
    hidden_dim = v1.size(2)
    assert v2.size(1) == num_vectors
    assert v2.size(2) == hidden_dim
    # Normalize the vectors along the hidden_dim dimension
    v1_norm = F.normalize(v1, p=2, dim=2)  # Shape: (batch_size, #vector, hidden_dim)
    v2_norm = F.normalize(v2, p=2, dim=2)  # Shape: (memory_size, #vector, hidden_dim)

    # Reshape or expand v1 and v2 for batch matrix multiplication
    v1_exp = v1_norm.unsqueeze(1).expand(batch_size, memory_size, num_vectors, hidden_dim)
    v2_exp = v2_norm.unsqueeze(0).expand(batch_size, memory_size, num_vectors, hidden_dim)

    # Compute cosine similarity
    cos_sim = torch.sum(v1_exp * v2_exp, dim=3)

    # cos_sim now has the shape: (batch_size, memory_size, #vector)
    assert cos_sim.size() == (batch_size, memory_size, num_vectors)
    return cos_sim


def my_l2_dist(v1, v2):
    """
    Compute L2 distance between two tensors

    Parameters
    ----------
    v1 : torch.Tensor
        Tensor of shape (batch_size, #vector, hidden_dim)
    v2 : torch.Tensor
        Tensor of shape (memory_size, #vector, hidden_dim)

    Returns
    -------
    l2_dist : torch.Tensor
        Tensor of shape (batch_size, memory_size, #vector)
    """

    batch_size = v1.size(0)
    memory_size = v2.size(0)
    num_vectors = v1.size(1)
    hidden_dim = v1.size(2)
    assert v2.size(1) == num_vectors
    assert v2.size(2) == hidden_dim
    # Reshape or expand v1 and v2 for batch matrix multiplication
    v1_exp = v1.unsqueeze(1).expand(batch_size, memory_size, num_vectors, hidden_dim)
    v2_exp = v2.unsqueeze(0).expand(batch_size, memory_size, num_vectors, hidden_dim)

    # Compute L2 distance
    l2_dist = torch.sum((v1_exp - v2_exp) ** 2, dim=3)

    # l2_dist now has the shape: (batch_size, memory_size, #vector)
    return l2_dist


def save_dataset(dataset: Dataset, path: str) -> None:
    """
    Save dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset to save
    path : str
        Path to save dataset
    """
    # mkdir
    os.makedirs(path, exist_ok=True)
    # subset
    if isinstance(dataset, Subset):
        torch.save(dataset.indices, os.path.join(path, "subset_indices.pt"))
        save_dataset(dataset.dataset, os.path.join(path, "dataset"))
    elif isinstance(dataset, datasets.Dataset):
        # huggingface dataset
        dataset.save_to_disk(path)
    elif isinstance(dataset, Dataset):
        # torch dataset
        torch.save(dataset, os.path.join(path, "dataset.pt"))
    else:
        torch.save(dataset, os.path.join(path, "dataset.pt"))


def load_dataset(path: str) -> Dataset:
    """
    Load dataset

    Parameters
    ----------
    path : str
        Path to load dataset
    """
    if os.path.exists(os.path.join(path, "subset_indices.pt")):
        indices = torch.load(os.path.join(path, "subset_indices.pt"))
        dataset = load_dataset(os.path.join(path, "dataset"))
        return Subset(dataset, indices)
    elif os.path.exists(os.path.join(path, "dataset.pt")):
        return torch.load(os.path.join(path, "dataset.pt"))
    else:
        return datasets.load_from_disk(path)


@dataclasses.dataclass
class EmbeddingRetrieverOutput:
    """
    Output of EmbeddingRetriever

    Parameters
    ----------
    embeddings : torch.Tensor
        Embeddings of the queries
    indices : torch.Tensor
        Indices of the retrieved examples
    distances : torch.Tensor
        Distances of the retrieved examples
    """

    distances: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    examples: Optional[list[dict[str, Any]]] = None
    example_embeddings: Optional[torch.Tensor] = None
    query_embeddings: Optional[torch.Tensor] = None
    retrieved_embeddings: Optional[torch.Tensor] = None


class EmbeddingRetriever(nn.Module):
    def __init__(
        self,
        database: Dataset,
        encoder: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        parent_tokenizer: transformers.PreTrainedTokenizerBase,
        span_keys: Optional[list[str]] = None,
        n: int = 1,
        distance: Literal["l2", "cos"] = "cos",
        sampling: Optional[int | float] = None,
        seed: int = 0,
        encoder_input_keys: list[str] = ["input_ids", "attention_mask"],
        add_distance_token: Optional[bool] = None,
        span_detect_function: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.database = database
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.span_keys = span_keys
        self.n = n
        self.distance = distance
        self.sampling = sampling
        self.seed = seed
        self.encoder_input_keys = encoder_input_keys
        self.add_distance_token = add_distance_token
        self.parent_tokenizer = parent_tokenizer
        self.span_detect_function = span_detect_function

        if self.span_keys is None:
            self.compute_vector_input_keys = self.encoder_input_keys
        else:
            self.compute_vector_input_keys = self.encoder_input_keys + self.span_keys

        self.rng = random.Random(self.seed)

        didx = list(range(len(self.database)))
        if self.sampling is not None:
            # shuffle index
            self.rng.shuffle(didx)
            if self.sampling > 1:
                didx = didx[: int(self.sampling)]
            elif self.sampling > 0:
                didx = didx[: int(len(self.database) * self.sampling)]
            else:
                raise ValueError(f"Invalid sampling: {self.sampling}")
            # sort
            didx = sorted(didx)
        self.database_index = {i: didx[i] for i in range(len(didx))}
        self.inv_database_index = {v: k for k, v in self.database_index.items()}
        self.didx = didx
        self.database = Subset(self.database, didx)

    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size

    def compute_vectors(self, encoder_inputs: dict[str, torch.Tensor] | transformers.BatchEncoding) -> torch.Tensor:
        if self.span_keys is None:
            # if span_keys is None, use cls token
            spans = torch.tensor(
                [[0, 1]] * len(encoder_inputs["input_ids"]), device=encoder_inputs["input_ids"].device
            ).unsqueeze(1)
        else:
            # spans.shape : (batch_size + self.max_sampling, len(self.span_keys), 2)
            spans = torch.stack([encoder_inputs[k] for k in self.span_keys], dim=1)

        inputs = {k: v for k, v in encoder_inputs.items() if k in self.encoder_input_keys}
        # emb.shape: (batch_size + self.max_sampling, sequence_length, vector_size)
        model_out = self.encoder(**inputs)
        emb = model_out.last_hidden_state

        # Expand dims of emb for broadcasting: new shape (batch_size, 1, sequence_length, vector_size)
        emb_expanded = emb.unsqueeze(1)

        # Create masks for spans
        seq_length = emb.size(1)
        mask = torch.arange(seq_length, device=emb.device)[None, None, :]  # shape: (1, 1, sequence_length)

        # Generate masks for each span: True for indices within the span, False otherwise
        # spans[..., 0] is start index, spans[..., 1] is end index
        mask = (mask >= spans[..., 0, None]) & (
            mask <= spans[..., 1, None]
        )  # shape: (batch_size, num_spans, sequence_length)

        # Use masked select to gather elements within the spans and sum them
        masked_emb = torch.where(mask[..., None], emb_expanded, torch.zeros_like(emb_expanded))

        # Compute the mean over the third dimension (sequence_length)
        span_lengths = (spans[..., 1] - spans[..., 0] + 1).float()  # Avoid integer division
        vectors = masked_emb.sum(dim=2) / span_lengths[..., None]
        return vectors

    def _prepare_inputs(self, examples: list[Any]) -> transformers.BatchEncoding:
        # filter encoder inputs
        examples = [{k: v for k, v in e.items() if k in self.compute_vector_input_keys} for e in examples]
        # examples = [{k: v for k, v in e.items() if k in self.encoder_input_keys} for e in examples]
        # concatenate using tokenizer with pad method
        encoder_inputs = self.tokenizer.pad(examples, return_tensors="pt", padding=True, pad_to_multiple_of=8)
        # add spans via span_detect_function
        # convert to the device of the model
        device = next(self.encoder.parameters()).device
        encoder_inputs.to(device)
        if self.span_detect_function is not None:
            spans: list[dict[str, torch.tensor]] = [self.span_detect_function(e) for e in encoder_inputs["input_ids"]]
            # concatenate
            spans = {k: torch.stack([s[k] for s in spans]).to(device) for k in spans[0].keys()}
            encoder_inputs.update(spans)
        return encoder_inputs

    def forward(self, *args, **kwargs) -> EmbeddingRetrieverOutput:
        raise NotImplementedError

    def update(self) -> None:
        pass

    def add_accelerator(self, accelerator: Optional[accelerate.Accelerator]):
        pass

    def get_update_callback(_self, update_step: Optional[int] = None) -> TrainerCallback:
        class UpdateEmbeddingCallback(TrainerCallback):
            def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
                if update_step is not None and (state.global_step + 1) % update_step == 0:
                    _self.update()

        return UpdateEmbeddingCallback()

    def compute_distance(
        self,
        query_emb: torch.Tensor,
        example_emb: torch.Tensor,
        reduction: str = "mean",
        combination: Optional[bool] = None,
    ):
        """
        Compute distance between query and examples

        Parameters
        ----------
        query_emb : torch.Tensor
            Embeddings of the queries
        example_emb : torch.Tensor
            Embeddings of the examples
        reduction : str
            Reduction method
        combination : bool
            If True, compute distance between query and examples for each span_keys and concatenate them

        Returns
        -------
        distance : torch.Tensor
            Distance between query and examples
        """
        assert len(query_emb.shape) == 3
        assert len(example_emb.shape) == 3
        assert query_emb.shape[1] == example_emb.shape[1]
        assert query_emb.shape[2] == example_emb.shape[2]

        if combination:
            # query_emb.shape: (batch_size, len(span_keys), vector_size)
            # example_emb.shape: (len(database), len(span_keys), vector_size)
            # to convert, expand second dimension to permutation with replacement
            # if len(span_keys) == 3, then pair_index = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), ...] : permutation with replacement
            pair_index = torch.tensor(
                list(itertools.product(range(query_emb.shape[1]), range(example_emb.shape[1]))),
                device=query_emb.device,
            )
            query_emb = query_emb[:, pair_index[:, 0]]
            example_emb = example_emb[:, pair_index[:, 1]]

        if self.distance == "l2":
            distance = my_l2_dist(query_emb, example_emb)
        elif self.distance == "cos":
            # cosine distance
            distance = 1 - my_cos_sim(query_emb, example_emb)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        # distance.shape: (batch_size, len(self.database), len(self.span_keys))
        assert distance.shape[0] == query_emb.shape[0]
        assert distance.shape[1] == example_emb.shape[0]
        # mean over span_keys
        if reduction == "mean":
            distance = distance.mean(dim=-1)
            assert distance.shape == (query_emb.shape[0], example_emb.shape[0])
        elif reduction == "max":
            distance = distance.max(dim=-1)[0]
            assert distance.shape == (query_emb.shape[0], example_emb.shape[0])
        elif reduction == "min":
            distance = distance.min(dim=-1)[0]
            assert distance.shape == (query_emb.shape[0], example_emb.shape[0])
        elif reduction == "softmax":
            weight = F.softmax(distance, dim=-1)
            distance = (distance * weight).sum(dim=-1)
            assert distance.shape == (query_emb.shape[0], example_emb.shape[0])
        elif reduction == "softmin":
            if self.distance == "l2":
                weight = F.softmin(distance, dim=-1)
            elif self.distance == "cos":
                weight = F.softmax(1 - distance, dim=-1)
            else:
                raise ValueError(f"Unknown distance metric: {self.distance}")
            distance = (distance * weight).sum(dim=-1)
            assert distance.shape == (query_emb.shape[0], example_emb.shape[0])
        elif reduction == "none":
            pass
        else:
            pass
        return distance

    @property
    def config(self):
        # arguments of __init__ except function args and database
        return {
            "span_keys": self.span_keys,
            "n": self.n,
            "distance": self.distance,
            "sampling": self.sampling,
            "seed": self.seed,
            "encoder_input_keys": self.encoder_input_keys,
            "add_distance_token": self.add_distance_token,
        }

    def save_pretrained(self, path: str):
        """
        Save model

        Parameters
        ----------
        path : str
            Path to save model
        """
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.parent_tokenizer.save_pretrained(os.path.join(path, "parent"))

        with open(f"{path}/model_config.json", "w") as f:
            json.dump(self.config, f)
        # save database
        save_dataset(self.database, os.path.join(path, "database"))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *args,
        legacy: Optional[bool] = None,
        **kwds,
    ):
        encoder = transformers.AutoModel.from_pretrained(model_name_or_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        parent_tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(model_name_or_path, "parent"))
        with open(f"{model_name_or_path}/model_config.json", "r") as f:
            config = json.load(f)
        config.update(kwds)
        if legacy:
            database = torch.load(f"{model_name_or_path}/database.pt")
        else:
            database = load_dataset(os.path.join(model_name_or_path, "database"))
        return cls(
            *args, database=database, encoder=encoder, tokenizer=tokenizer, parent_tokenizer=parent_tokenizer, **config
        )

    def forward_input_to_base_model_input(self, input_ids: torch.Tensor, *args, **kwds) -> transformers.BatchEncoding:
        """Convert forward input to base model input"""
        # decode input_ids to string
        string_inputs = self.parent_tokenizer.batch_decode(input_ids)
        # tokenize
        base_model_inputs = self.tokenizer(string_inputs, return_tensors="pt", padding=True, pad_to_multiple_of=8)
        return base_model_inputs


class EmbeddingRetrieverWithStore(EmbeddingRetriever):
    """
    Embedding retriever with store

    Parameters
    ----------
    database : SupportsIndex
        Database to retrieve
    """

    def __init__(
        self,
        *args,
        verbose: bool = True,
        collator: Optional[Callable] = None,
        update_batch_size: int = 1,
        accelerator: Optional[accelerate.Accelerator] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.update_batch_size = update_batch_size
        self.verbose = verbose
        if collator is None:
            self.collator = DataCollatorWithPadding(self.tokenizer)
        else:
            self.collator = collator
        self.vectors = None
        # select column for update (["input_ids", "attention_mask"])
        # database_for_update = Subset(self.database.dataset.select_columns(["input_ids", "attention_mask"]), self.didx)
        # This doesn't work for torch dataset
        # select column for update (["input_ids", "attention_mask"])

        def col(x):
            x = [{k: v for k, v in e.items() if k in self.compute_vector_input_keys} for e in x]
            return self.collator(x)

        self.database_loader = DataLoader(
            self.database,
            batch_size=self.update_batch_size,
            collate_fn=col,
            shuffle=False,
        )
        self.add_accelerator(accelerator)

    @property
    def config(self):
        return {**super().config, "update_batch_size": self.update_batch_size}

    def add_accelerator(self, accelerator: Optional[accelerate.Accelerator]):
        self.accelerator = accelerator
        if accelerator is not None:
            damy_optim = torch.optim.SGD(self.encoder.parameters(), lr=1e-3)
            self.encoder, self.loader, _ = accelerator.prepare(self.encoder, self.database_loader, damy_optim)

    @torch.no_grad()
    def update(self) -> None:
        """
        Compute vectors for all examples in database
        """
        training = self.encoder.encoder.training
        self.encoder.eval()
        # shape: (len(self.database), len(self.span_keys), vector_size)
        num_vect = len(self.span_keys) if self.span_keys is not None else 1
        self.vectors = torch.Tensor(len(self.database), num_vect, self.encoder.config.hidden_size)
        # concatenate using tokenizer with pad method
        loader = self.database_loader
        idx = 0
        device = next(self.encoder.parameters()).device if self.accelerator is None else self.accelerator.device
        for batch in tqdm(loader, desc="Updating vectors", disable=not self.verbose):
            batch = batch.to(device)
            # detect spans
            if self.span_detect_function is not None:
                spans: list[dict[str, torch.tensor]] = [self.span_detect_function(e) for e in batch["input_ids"]]
                # concatenate
                spans = {k: torch.stack([s[k] for s in spans]).to(device) for k in spans[0].keys()}
                batch.update(spans)
            # compute vectors
            vectors = self.compute_vectors(batch)
            # vectors.shape: (batch_size, len(self.span_keys), vector_size)
            # set vectors
            if self.accelerator is None:
                self.vectors[idx : idx + len(vectors)] = vectors
                idx += len(vectors)
            else:
                vectors = self.accelerator.gather(vectors)
                if self.accelerator.is_local_main_process:
                    self.vectors[idx : idx + len(vectors)] = vectors
                    idx += len(vectors)
        # broadcast to other processes
        if self.accelerator is not None:
            self.vectors = accelerate.utils.broadcast(self.vectors)

        self.encoder.train(training)

    def get_update_callback(_self, update_step: Optional[int] = None) -> TrainerCallback:
        class UpdateEmbeddingCallback(TrainerCallback):
            # def on_evaluate_begin(self, args, state, control, **kwargs):
            #     _self.update()

            def on_step_begin(self, args, state, control, **kwargs):
                if update_step is not None and (state.global_step + 1) % update_step == 0:
                    _self.update()

        return UpdateEmbeddingCallback()

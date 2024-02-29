from typing import Optional

import torch
from torch.utils.data import RandomSampler

from .base import EmbeddingRetrieverOutput, EmbeddingRetrieverWithStore


class KNNEmbeddingRetriever(EmbeddingRetrieverWithStore):
    """
    k-nearest neighbor retriever

    Parameters
    ----------
    database : SupportsIndex
        Database to retrieve
    """

    def __init__(
        self,
        *args,
        max_sampling: Optional[int] = None,
        top_k: Optional[int] = None,
        differentiable: Optional[bool] = None,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        distance_reduction: str = "mean",
        distance_combination: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.max_sampling = max_sampling
        self.top_k = top_k
        self.differentiable = differentiable
        self.temperature = temperature
        self.epsilon = epsilon
        self.distance_reduction = distance_reduction
        self.distance_combination = distance_combination

        # max_sampling and top_k cannot specify at the same time
        assert not (
            self.max_sampling is not None and self.top_k is not None
        ), "max_sampling and top_k cannot specify at the same time"
        # each example in database must have: input_ids, attention_mask, token_type_ids
        if self.max_sampling is None:
            self.max_sampling = self.n
        assert self.max_sampling >= self.n, f"max_sampling must be greater than or equal to n"

    @property
    def config(self):
        super_config = super().config
        super_config.update(
            {
                "max_sampling": self.max_sampling,
                "top_k": self.top_k,
                "differentiable": self.differentiable,
                "temperature": self.temperature,
                "epsilon": self.epsilon,
            }
        )
        return super_config

    def select_topn(self, vectors: torch.Tensor, distance: torch.Tensor, n: int) -> torch.Tensor:
        """
        Select top-n examples from database

        Parameters
        ----------
        vectors : torch.Tensor
            Vectors to compute distance
        distance : torch.Tensor
            Distance between vectors and database
        n : int
            Number of examples to select

        Returns
        -------
        torch.Tensor
            Selected indices
        """
        # distance.shape: (batch_size, len(self.database))
        # indices.shape: (batch_size, n)
        indices = distance.argsort(dim=-1, descending=False)[:, :n]
        # distance.shape: (batch_size, n)
        distance = distance.gather(dim=-1, index=indices)

        return indices, distance

    def forward(
        self,
        input_ids: torch.Tensor,
        n: Optional[int] = None,
        retriever_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EmbeddingRetrieverOutput:
        # TODO given indices
        if n is None:
            n = self.n
        batch_size = input_ids.shape[0]
        base_input = self.forward_input_to_base_model_input(input_ids, **kwargs)
        kwargs.update(base_input)
        kwargs = {k: v for k, v in kwargs.items() if k in self.compute_vector_input_keys}
        example_list = [{k: v[i] for k, v in kwargs.items()} for i in range(len(kwargs["input_ids"]))]
        # example_list_with_header = [{self.header + k: v for k, v in e.items()} for e in example_list]
        example_list_with_header = example_list
        if retriever_indices is None:
            if self.training:
                if self.top_k is not None:
                    # train on top_k examples
                    assert self.vectors is not None, "vectors must be computed before inference. Call update() first."
                    # compute query_emb
                    # query_emb.shape: (batch_size, len(self.span_keys), vector_size)
                    query_inputs = self._prepare_inputs(example_list_with_header)
                    query_emb = self.compute_vectors(query_inputs.to(device=input_ids.device))

                    # compute distance
                    # distance.shape: (batch_size, len(self.database), len(self.span_keys))
                    distance = self.compute_distance(
                        query_emb.to(self.vectors),
                        self.vectors,
                        reduction=self.distance_reduction,
                        combination=self.distance_combination,
                    )
                    assert distance.shape == (batch_size, len(self.database))

                    # sample top-k indices
                    # indices.shape: (batch_size, self.top_k)
                    topk_indices = distance.argsort(dim=-1, descending=False)[:, : self.top_k]
                    # random sampling self.n examples from top-k
                    example_indices = []
                    for i in range(batch_size):
                        example_indices.append(self.rng.sample(topk_indices[i].tolist(), self.n))
                    # example_indices.shape: (batch_size, self.n)
                    example_indices = torch.tensor(example_indices)

                    # epsilon-greedy
                    if self.epsilon > 0:
                        random_mask = (
                            torch.zeros_like(example_indices, dtype=torch.float).fill_(self.epsilon).bernoulli().bool()
                        )
                        # sample random indices
                        random_indices = torch.randint(
                            low=0, high=len(self.database), size=(batch_size, self.n), device=example_indices.device
                        )
                        # replace indices with random indices
                        example_indices = torch.where(random_mask, random_indices, example_indices)

                    # recompute vectors for sampled examples
                    uniq_idx = example_indices.flatten().unique().tolist()

                    examples = list(self.database[i] for i in uniq_idx)
                    encoder_inputs = self._prepare_inputs(examples)
                    # example_emb.shape: (len(uniq_idx), len(self.span_keys), vector_size)
                    example_emb = self.compute_vectors(encoder_inputs.to(input_ids.device))

                    mapping = {i: idx for idx, i in enumerate(uniq_idx)}

                    distance = self.compute_distance(
                        query_emb,
                        example_emb,
                        reduction=self.distance_reduction,
                        combination=self.distance_combination,
                    )

                    # map example_indices to uniq_idx
                    example_indices = torch.tensor([[mapping[i.item()] for i in idx] for idx in example_indices]).to(
                        input_ids.device
                    )

                    # indices.shape: (batch_size, n)
                    # create indices by resorting example_indices based on actual distance
                    indices = torch.zeros(batch_size, n, dtype=torch.long, device=input_ids.device)
                    for i, idx in enumerate(example_indices):
                        indices[i] = example_indices[i][distance[i][idx].argsort(dim=-1, descending=False)]
                    # indices.shape: (batch_size, n)
                    # indices = example_indices.gather(dim=-1, index=distance.argsort(dim=-1, descending=False))
                    # distance.shape: (batch_size, n)
                    # distance = distance.gather(dim=-1, index=indices)
                else:
                    # random sampling self.max_sampling examples from database
                    random_sampler = RandomSampler(self.database, replacement=True, num_samples=self.max_sampling)
                    # train on random sampling
                    example_indices = [i for i in random_sampler]
                    # get examples
                    examples = list(self.database[i] for i in example_indices)

                    # concatenate
                    ex_list = example_list_with_header + examples
                    encoder_inputs = self._prepare_inputs(ex_list)

                    # compute vectors
                    # vectors.shape: (batch_size + self.max_sampling, len(self.span_keys), vector_size)
                    vectors = self.compute_vectors(encoder_inputs.to(input_ids.device))

                    query_emb = vectors[:batch_size]
                    example_emb = vectors[batch_size:]

                    # compute distance
                    # distance.shape: (batch_size, self.max_sampling, len(self.span_keys))
                    distance = self.compute_distance(
                        query_emb,
                        example_emb,
                        reduction=self.distance_reduction,
                        combination=self.distance_combination,
                    )
                    # indices.shape: (batch_size, self.max_sampling)
                    indices = distance.argsort(dim=-1, descending=False)
                    # indices.shape: (batch_size, n)
                    indices = indices[:, :n]
                    # distance.shape: (batch_size, n)
                    # distance = distance.gather(dim=-1, index=indices)
            else:
                # find nearest neighbors from stored vectors
                # compute vectors for queries
                encoder_inputs = self._prepare_inputs(example_list_with_header)
                query_emb = self.compute_vectors(encoder_inputs)
                # query_emb.shape: (batch_size, len(self.span_keys), vector_size)
                # compute distance
                # distance.shape: (batch_size, len(self.database))
                assert self.vectors is not None, "vectors must be computed before inference. Call update() first."
                distance = self.compute_distance(
                    query_emb.to(self.vectors),
                    self.vectors,
                    reduction=self.distance_reduction,
                    combination=self.distance_combination,
                )
                assert distance.shape == (batch_size, len(self.database))
                # indices.shape: (batch_size, n)
                indices = distance.argsort(dim=-1, descending=False)[:, :n]
                example_emb = self.vectors
                examples = self.database

            # select example
            if self.differentiable:
                # compute distribution
                # distribution.shape: (batch_size, n, len(self.database))
                distribution = self.compute_knn_distribution(-distance, n, temperature=self.temperature)
                # multiply distribution with example_emb and sum over database
                # distribution.shape: (batch_size, n, len(self.database))
                # example_emb.shape: (len(self.database), len(self.span_keys), vector_size)
                # retrieved_embeddings.shape: (batch_size, n, len(self.span_keys), vector_size)
                retrieved_embeddings = torch.einsum("ijk,klm->ijlm", distribution, example_emb)
                num_vect = len(self.span_keys) if self.span_keys is not None else 1
                assert retrieved_embeddings.shape == (batch_size, n, num_vect, self.hidden_size)
            else:
                retrieved_embeddings = example_emb[indices.flatten()].view(batch_size, n, len(self.span_keys), -1)
                if self.add_distance_token:
                    # add distance token to the end of each example in retrieved_embeddings
                    dist = distance.gather(dim=-1, index=indices[:, :n])
                    retrieved_embeddings = torch.cat(
                        [retrieved_embeddings, dist.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)],
                        dim=-2,
                    )
        else:
            # indices.shape: (batch_size, n)
            indices = retriever_indices
            # examples.shape: (batch_size, n)
            examples = list(self.database[i] for i in indices.flatten().tolist())
            # concatenate
            ex_list = example_list_with_header + examples
            encoder_inputs = self._prepare_inputs(ex_list)
            # compute vectors
            # vectors.shape: (batch_size + n, len(self.span_keys), vector_size)
            vectors = self.compute_vectors(encoder_inputs.to(input_ids.device))
            query_emb = vectors[:batch_size]
            example_emb = vectors[batch_size:]
            # compute distance
            # distance.shape: (batch_size, n, len(self.span_keys))
            distance = self.compute_distance(
                query_emb,
                example_emb,
                reduction=self.distance_reduction,
                combination=self.distance_combination,
            )
            # retrieved_embeddings.shape: (batch_size, n, len(self.span_keys), vector_size)
            retrieved_embeddings = example_emb
            if self.add_distance_token:
                # add distance token to the end of each example in retrieved_embeddings
                retrieved_embeddings = torch.cat(
                    [retrieved_embeddings, distance.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)],
                    dim=-2,
                )

        return EmbeddingRetrieverOutput(
            indices=indices,
            distances=distance,
            examples=examples,
            example_embeddings=example_emb.to(query_emb),
            query_embeddings=query_emb,
            retrieved_embeddings=retrieved_embeddings.to(query_emb),
        )

    @staticmethod
    def compute_knn_distribution(scores: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute distribution of scores

        Parameters
        ----------
        scores : torch.Tensor
            Scores to compute distribution
            shape: (batch_size, num_examples)

        Returns
        -------
        torch.Tensor
            Distribution of scores
            shape: (batch_size, k, num_examples)
        """
        assert k >= 1, "k must be greater than or equal to 1"
        # scores.shape: (batch_size, num_examples)
        distribution = torch.zeros(scores.shape[0], k, scores.shape[1], device=scores.device)
        # distribution.shape: (batch_size, k, num_examples)
        for i in range(k):
            distribution[:, i, :] = (scores / temperature).softmax(dim=-1)
            scores = scores + torch.log(1 - distribution[:, i, :])

        return distribution

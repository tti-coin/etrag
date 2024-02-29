from typing import Optional

import torch

from .base import EmbeddingRetriever, EmbeddingRetrieverOutput


class RandomEmbeddingRetriever(EmbeddingRetriever):
    """
    Random retriever

    Parameters
    ----------
    database : SupportsIndex
        Database to retrieve
    """

    def forward(self, input_ids: torch.Tensor, *args, n: Optional[int] = None, **kwargs) -> EmbeddingRetrieverOutput:
        if n is None:
            n = self.n
        batch_size = input_ids.shape[0]
        base_input = self.forward_input_to_base_model_input(input_ids, **kwargs)
        kwargs.update(base_input)
        # kwargs["attention_mask"] = attention_mask
        kwargs = {k: v for k, v in kwargs.items() if k in self.compute_vector_input_keys}
        example_list = [{k: v[i] for k, v in kwargs.items()} for i in range(len(kwargs["input_ids"]))]
        example_list_with_header = example_list

        # random sampling self.n examples from database
        example_indices = self.rng.sample(range(len(self.database)), self.n)
        examples = list(self.database[i] for i in example_indices)
        # # concatenate
        # encoder_inputs = self._prepare_inputs(examples)
        # # compute vectors
        # # vectors.shape: (self.n, len(self.span_keys), vector_size)
        # vectors = self.compute_vectors(encoder_inputs.to(input_ids.device))
        # # query_emb.shape: (batch_size, len(self.span_keys), vector_size)
        # example_emb = self.compute_vectors(kwargs)

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
        distance = self.compute_distance(query_emb, example_emb)
        # indices.shape: (batch_size, n)
        indices = distance.argsort(dim=-1, descending=False)

        # select example
        retrieved_embeddings = example_emb[indices.flatten()].view(batch_size, n, len(self.span_keys), -1)

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

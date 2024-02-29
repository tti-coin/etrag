import collections
import dataclasses
import os
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import peft
import safetensors.torch
import torch
import transformers
from torch import nn

from .retriever import (
    EmbeddingRetriever,
    EmbeddingRetrieverOutput,
    KNNEmbeddingRetriever,
)


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


def load_pretrained(model, directory, load_state_dict=True):
    if load_state_dict:
        if isinstance(model, peft.PeftModel):
            model.load_adapter(directory, model.active_adapter)
        else:
            try:
                safetensors.torch.load_model(model, os.path.join(directory, "model.safetensors"))
            except:
                state_dict = torch.load(os.path.join(directory, "pytorch_model.bin"))
                model.load_state_dict(state_dict)
    else:
        model = model.from_pretrained(directory)
    return model


class Seq2SeqModelWithRetriever(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        retriever: EmbeddingRetriever,
        *args,
        insert: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.retriever: EmbeddingRetriever = retriever
        for named_param, value in list(base_model.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = base_model.get_submodule(named_param.replace(".weight", ""))
                break
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )
        self.projection = nn.Linear(self.retriever.hidden_size, self.base_model.config.hidden_size)
        self.insert = insert

    def get_prompt(self, *args, return_retriever_outputs: Optional[bool] = None, **kwargs):
        # generate prompt
        retriever_output: EmbeddingRetrieverOutput = self.retriever(*args, **kwargs)
        assert retriever_output.query_embeddings is not None, "Retriever output must have query embeddings"
        assert retriever_output.retrieved_embeddings is not None, "Retriever output must have retrieved embeddings"
        # query_embeddings: (batch_size, num_vector, hidden_size)
        # retrieved_embeddings: (batch_size, retrieval_size, num_vector, hidden_size)
        # embeddings: (batch_size, retrieval_size + 1, num_vector, hidden_size)
        batch_size = retriever_output.retrieved_embeddings.shape[0]
        hidden_dim = retriever_output.retrieved_embeddings.shape[-1]
        prompts = torch.cat(
            (retriever_output.query_embeddings, retriever_output.retrieved_embeddings.view(batch_size, -1, hidden_dim)),
            dim=1,
        )
        # prompts: (batch_size, (retrieval_size + 1) * num_vector, hidden_size)
        # prompts = embeddings.view(embeddings.shape[0], -1, embeddings.shape[-1])
        # move device
        self.projection = self.projection.to(prompts.device)

        prompts = self.projection(prompts)
        if return_retriever_outputs:
            return prompts, retriever_output
        return prompts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        retriever_indices=None,
        prompts=None,
        **kwargs,
    ):
        # generate prompt
        retriever_output = None
        if prompts is None:
            prompts, retriever_output = self.get_prompt(
                input_ids=input_ids, retriever_indices=None, return_retriever_outputs=True
            )
        batch_size = _get_batch_size(input_ids, inputs_embeds)

        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).to(decoder_attention_mask.device)
            decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None

        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).to(attention_mask.device)
            if self.insert:
                # insert after cls token
                attention_mask = torch.cat((attention_mask[:, :1], prefix_attention_mask, attention_mask[:, 1:]), dim=1)
            else:
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # kwargs.update(
        #     {
        #         "attention_mask": attention_mask,
        #         "decoder_attention_mask": decoder_attention_mask,
        #         "labels": labels,
        #         "output_attentions": output_attentions,
        #         "output_hidden_states": output_hidden_states,
        #         "return_dict": return_dict,
        #     }
        # )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if self.insert:
            inputs_embeds = torch.cat((inputs_embeds[:, :1], prompts, inputs_embeds[:, 1:]), dim=1)
        else:
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        model_input = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_inputs_embeds": decoder_inputs_embeds,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }

        out = self.base_model(**model_input)
        out["retriever_output"] = retriever_output
        out["prompts"] = prompts
        return out

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if "input_ids" not in kwargs:
                raise ValueError("input_ids must be provided for Peft model generation")
            if kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                kwargs["position_ids"] = None
            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            kwargs = deepcopy(kwargs)

            if "encoder_outputs" in kwargs:
                del kwargs["encoder_ouputs"]
                warnings.warn(
                    "`encoder_outputs` should not be passed to `generate` when using prompt tuning. Ignoring it."
                )

            input_ids = kwargs.pop("input_ids")
            inputs_embeds = self.word_embeddings(input_ids)
            batch_size = inputs_embeds.shape[0]

            retriever_input_keys = {
                k: k[len(self.retriever_header) :] for k in kwargs if k.startswith(self.retriever_header)
            }
            retriever_input = {retriever_input_keys[k]: kwargs.pop(k) for k in retriever_input_keys}
            # generate prompt
            prompts = self.get_prompt(**retriever_input)

            inputs_embeds = torch.cat((prompts[:, : prompts.shape[1]], inputs_embeds), dim=1)
            kwargs["inputs_embeds"] = inputs_embeds

            if "attention_mask" in kwargs:
                prefix_attention_mask = torch.ones(batch_size, prompts.shape[1]).to(kwargs["attention_mask"].device)
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

            # pop database keys
            database_input_keys = {
                k: k[len(self.database_header) :] for k in kwargs if k.startswith(self.database_header)
            }
            database_input = {database_input_keys[k]: kwargs.pop(k) for k in database_input_keys}

            return self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        return model_kwargs

    def save_pretrained(self, save_directory, *args, **kwds):
        self.base_model.config.save_pretrained(save_directory, *args, **kwds)
        self.base_model.save_pretrained(save_directory, *args, **kwds)
        # save retriever
        ret_dir = os.path.join(save_directory, "retriever_model")
        self.retriever.save_pretrained(ret_dir)
        # save projection state dict
        torch.save(self.projection.state_dict(), os.path.join(save_directory, "projection.pt"))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        *args,
        retriever_class=KNNEmbeddingRetriever,
        retriever_kwargs: dict[str, Any] = {},
        insert: Optional[bool] = None,
        **kwds,
    ):
        try:
            base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, *args, **kwds)
        except:
            config = transformers.AutoConfig.from_pretrained(model_name_or_path, *args, **kwds)
            m = transformers.AutoModelForSeq2SeqLM.from_config(config)
            try:
                import peft

                base_model = peft.PeftModel.from_pretrained(m, model_id=model_name_or_path)
            except:
                base_model = m

        sd = os.path.join(model_name_or_path, "retriever_model")
        retriever = retriever_class.from_pretrained(sd, **retriever_kwargs)
        # load projection
        projection_state_dict = torch.load(os.path.join(model_name_or_path, "projection.pt"), map_location="cpu")
        model = cls(base_model, retriever, *args, insert=insert, **kwds)
        model.projection.load_state_dict(projection_state_dict)
        return model

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        return self.base_model._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwds):
        self.base_model.gradient_checkpointing_enable(*args, **kwds)
        if hasattr(self.retriever, "encoder") and hasattr(self.retriever.encoder, "gradient_checkpointing_enable"):
            self.retriever.encoder.gradient_checkpointing_enable(*args, **kwds)

    def gradient_checkpointing_disable(self, *args, **kwds):
        self.base_model.gradient_checkpointing_disable(*args, **kwds)
        if hasattr(self.retriever, "encoder") and hasattr(self.retriever.encoder, "gradient_checkpointing_disable"):
            self.retriever.encoder.gradient_checkpointing_disable(*args, **kwds)

    def update_retriever(self):
        if hasattr(self.retriever, "update"):
            self.retriever.update()

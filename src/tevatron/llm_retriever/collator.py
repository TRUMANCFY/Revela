import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from tevatron.llm_retriever.arguments import DataArguments
import copy
import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=False, 
            truncation=True,
            max_length=self.data_args.passage_max_len-1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]
        
        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return q_collated, d_collated


@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_texts = self.tokenizer(
            texts,
            padding=False, 
            truncation=True,
            max_length=max_length-1 if self.data_args.append_eos_token else max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_texts['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_texts['input_ids']]
        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return text_ids, collated_texts


@dataclass
class LLMEnhancedCollator:
    data_args: DataArguments
    retriever_tokenizer: PreTrainedTokenizer # Tokenizer for the retriever (LLaMA)
    llm_tokenizer: PreTrainedTokenizer # Always LLaMA tokenizer for LLM processing
    encoder_type: str # "llama"

    
    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        data_src = features[0][0][0]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])

        # Tokenizer setting for passages (retriever)
        retriever_tokenizer_kwargs = {
            "padding": False,
            "truncation": True,
            "max_length": self.data_args.passage_max_len - 1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
            "return_attention_mask": True,
            "return_token_type_ids": False,
            "add_special_tokens": True,
        }

        # Tokenizer setting for passages (LLM)
        llm_tokenizer_kwargs = retriever_tokenizer_kwargs.copy()

        # Toknize passages using the retriever tokenizer
        d_collated = self.retriever_tokenizer(all_passages, **retriever_tokenizer_kwargs)

        # Tokenize for LLM
        d_collated_llm = self.llm_tokenizer(all_passages, **llm_tokenizer_kwargs)

        # Apply passage prefix
        if self.data_args.add_passage_prefix:
            all_passages_passage_prefix = [self.data_args.passage_prefix + p for p in all_passages]
            d_collated_passage = self.retriever_tokenizer(all_passages_passage_prefix, **retriever_tokenizer_kwargs)
        else:
            d_collated_passage = copy.deepcopy(d_collated)
        
        # Apply query prefix
        if self.data_args.add_query_prefix:
            all_passages_query_prefix = [self.data_args.query_prefix + p for p in all_passages]
            d_collated_query = self.retriever_tokenizer(all_passages_query_prefix, **retriever_tokenizer_kwargs)
        else:
            d_collated_query = copy.deepcopy(d_collated)

        # Append EOS token for LLaMA
        if self.data_args.append_eos_token:
            d_collated_passage['input_ids'] = [d + [self.retriever_tokenizer.eos_token_id] for d in d_collated_passage['input_ids']]
            d_collated_query['input_ids'] = [d + [self.retriever_tokenizer.eos_token_id] for d in d_collated_query['input_ids']]

        # Padding settings based on encoder type
        pad_kwargs = {
            "padding": "max_length",
            "pad_to_multiple_of": self.data_args.pad_to_multiple_of,
            "return_attention_mask": True,
            "return_tensors": "pt",
            "max_length": self.data_args.passage_max_len,
        }

        # Pad passage inputs
        d_collated_passage_padded = self.retriever_tokenizer.pad({"input_ids": d_collated_passage['input_ids']}, **pad_kwargs)

        # LLM tokenization always uses the LlaMA tokenizer
        d_collated_llm = self.llm_tokenizer.pad(d_collated_llm, **pad_kwargs)

        # Adjust labels for LLaMA (set pad_token_id to -100)
        labels = d_collated_llm["input_ids"].clone()
        if self.llm_tokenizer.pad_token_id is not None:
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
        d_collated_llm["labels"] = labels


        # first half processing
        # avoid stackoverflow to be used for first half
        if self.data_args.first_half and data_src != 'stackoverflow':
            #print('continue with ', data_src)
            d_collated_query_curated = {}
            input_ids = d_collated_query["input_ids"]
            attention_mask = d_collated_query["attention_mask"]

            new_input_ids = []
            new_attention_mask = []

            for ids, mask in zip(input_ids, attention_mask):
                eos_token_id = self.retriever_tokenizer.eos_token_id
                if ids[-1] == eos_token_id:
                    ids = ids[:-1]
                
                prefix_len = len(ids) - len(mask)
                half_len = int(len(ids) * self.data_args.first_half_ratio)
                first_half = ids[:half_len]                
                
                first_half.append(eos_token_id)

                new_input_ids.append(first_half)
                first_half_mask = mask[:half_len - prefix_len]
                new_attention_mask.append(first_half_mask)
            
            # Pad new attention mask
            new_attention_mask_padded = [
                mask + [0] * (d_collated_llm['input_ids'].shape[1] - len(mask)) for mask in new_attention_mask
            ]

            d_collated_query_curated["input_ids"] = new_input_ids
            d_collated_query = d_collated_query_curated
            d_collated_llm["first_half_mask"] = torch.tensor(new_attention_mask_padded, dtype=torch.long)

        # Pad query inputs
        d_collated_query_padded = self.retriever_tokenizer.pad(
            {"input_ids": d_collated_query['input_ids']},
            **pad_kwargs
        )

        return d_collated_passage_padded, d_collated_llm, d_collated_query_padded

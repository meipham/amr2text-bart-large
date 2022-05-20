# coding:utf-8
import os
import torch
import inspect
import importlib
import pytorch_lightning as pl
from datasets import load_dataset
from dataclasses import dataclass
import transformers
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
from torch.utils.data.dataloader import DataLoader
from common.utils import shift_tokens_right
# from transformers.data.data_collator import default_data_collator, DataCollatorForSeq2Seq 
        
class AMR2TextDataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer, **args,
    ):
        super().__init__()
        self.train_file = args["train_data_file"]
        self.validation_file = args["eval_data_file"] 
        self.test_file = args["test_data_file"] 
        self.src_prefix = args["src_prefix"] 
        self.tgt_prefix = args["tgt_prefix"]
        self.pad_to_max_length = False
        self.ignore_pad_token_for_loss = True
        self.cache_dir = args["cache_dir"]
        self.unified_inp = args["unified_input"]
        self.train_batchsize = args["per_gpu_train_batch_size"]
        self.val_batchsize = args["per_gpu_eval_batch_size"]
        self.train_num_worker = args["train_num_workers"]
        self.val_num_worker = args["eval_num_workers"]
        self.preprocess_worker = args["process_num_workers"]
        self.tokenizer = tokenizer
        self.max_sent_length = min(args["src_block_size"], self.tokenizer.model_max_length)
        self.max_amr_length = min(args["tgt_block_size"], self.tokenizer.model_max_length)

        # self.collate_fn = default_data_collator
        self.collate_fn = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if args["fp16"] else None
        )

    def setup(self, stage="fit"):
        data_files = {}
        if self.train_file: data_files["train"] = self.train_file
        if self.validation_file: data_files["validation"] = self.validation_file
        if self.test_file: data_files["test"] = self.test_file

        print("Dataset cache dir:", self.cache_dir)

        amr2textdatasets = load_dataset(
            f"{os.path.dirname(__file__)}/data.py", data_files=data_files, cache_dir=self.cache_dir,
        )
        
        print("datasets:", amr2textdatasets)
        column_names = amr2textdatasets["train"].column_names
        print("colums:", column_names)

        def preprocess_function(examples):
            
            tokenizer = self.tokenizer
            
            inputs = examples['tgt'] 
            targets = examples['src'] 
            
            model_inputs = tokenizer(inputs, 
                                     max_length=self.max_amr_length, 
                                     padding='max_length', 
                                     truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, 
                                   max_length=self.max_sent_length, 
                                   add_special_tokens = False, 
                                   padding='max_length', 
                                   truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            # labels["input_ids"] = [
            #     [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            # ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Train dataset
        self.train_dataset = amr2textdatasets['train'].map(
            preprocess_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=['src', 'tgt'],
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )
        print(f"ALL {len(self.train_dataset)} training instances")
        print("Test Dataset Instance Example:", self.train_dataset[0])
        
        # Valid dataset
        self.valid_dataset = amr2textdatasets['validation'].map(
            preprocess_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=['src', 'tgt'],
            load_from_cache_file=False,
            desc="Running tokenizer on valid_dataset dataset",
        )
        print(f"ALL {len(self.valid_dataset)} validation instances")
        print("Validation Dataset Instance Example:", self.valid_dataset[0])
        
        self.test_dataset = amr2textdatasets['test'].map(
            preprocess_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=['src', 'tgt'],
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset",
        )
        print(f"ALL {len(self.test_dataset)} testing instances")
        print("Test Dataset Instance Example:", self.test_dataset[0])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batchsize,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.train_num_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.val_batchsize,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.val_num_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batchsize,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.val_num_worker,
            pin_memory=True,
        )

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}"
            )

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    decoder_start_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features):

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        attention_mask = features["input_ids"].ne(self.tokenizer.pad_token_id).int()
        # prepare decoder_input_ids

        features["decoder_input_ids"] = shift_tokens_right(
            features["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

        return {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
            "labels": features["labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }
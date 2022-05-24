import argparse
import glob
import os
import pickle

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from spring_amr.tokenization_bart import PENMANBartTokenizer
from transformers import AutoConfig

from common.callbacks import (LoggingCallback, get_checkpoint_callback,
                              get_early_stopping_callback)
from common.options import add_model_specific_args
from data_interface.dataset_pl import AMR2TextDataModule
from model_interface.model_amr2text import AMR2TextModelModule
from datasets import load_metric
from transformers import (
    AutoConfig,
    MBartForConditionalGeneration,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
    )

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            
def load_callbacks(args, model):
    callbacks = []
    callbacks.append(LoggingCallback())

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
        callbacks.append(es_callback)

    lower_is_better = args.val_metric == "loss"
    checkpoint_callback = get_checkpoint_callback(
        args.output_dir,
        model.val_metric,
        save_top_k=args.save_total_limit,
        lower_is_better=lower_is_better,
        save_interval=args.save_interval,
    )
    callbacks.append(checkpoint_callback)

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval="step"))

    callbacks.append(CheckpointEveryNSteps(1000))

    return callbacks

def _load_data(tokenizer, args): 
    data_module = AMR2TextDataModule(tokenizer, **vars(args))
    data_module.setup()
    args.train_dataset_size = len(data_module.train_dataset)

    return data_module, args

def _load_tokenizer(args):
    tokenizer_name_or_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path
    amr_tokenizer = PENMANBartTokenizer.from_pretrained(
        tokenizer_name_or_path,
        collapse_name_ops=False,
        use_pointer_tokens=True,
        raw_graph=False,
        config = AutoConfig.from_pretrained(tokenizer_name_or_path)
    )
    return amr_tokenizer
    
def _load_model(load_path, tokenizer, args):
    if load_path is None:
        model = AMR2TextModelModule(tokenizer, args)
    else:                                   # to test
        model = AMR2TextModelModule(tokenizer, args)
        args.resume_from_checkpoint = load_path

    print(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.model.parameters()),
            sum(p.numel() for p in model.model.parameters() if p.requires_grad),
        )
    )
    return model

def _train(model, data_module, args):
    logger = TensorBoardLogger(save_dir="exp_log", name=args.output_dir)
    args.logger = logger
    args.callbacks = load_callbacks(args, model)

    train_params = {}
    if args.fp16:
        train_params["precision"] = 16
        # train_params["amp_backend"] = "apex"
        # train_params["amp_level"] = args.fp16_opt_level
    if args.gpus != None:
        train_params["accelerator"] = "gpu"
        train_params["strategy"] = "ddp"
    
    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    train_params["deterministic"] = True
    trainer = Trainer.from_argparse_args(args, **train_params)

    if args.do_train:
        print("Start Training ...")
        trainer.fit(model, datamodule=data_module)
    
    if args.do_predict:
        print("Predict on test Set ...")
        trainer.predict(model, dataloaders=data_module.test_dataloader())
    
    return model

def main(args):
    pl.seed_everything(args.seed)
    odir = Path(args.output_dir)
    odir.mkdir(exist_ok=True)
    load_path = None
    
    if args.resume:
        checkpoints = list(
            sorted(glob.glob(os.path.join(args.output_dir, "last.ckpt"), recursive=True))
        )
        assert (
            len(checkpoints) >= 1
        ), f"No checkpoints founded at {os.path.join(args.output_dir, 'last.ckpt')}"
        load_path = checkpoints[-1]

    amr_tokenizer = _load_tokenizer(args)
    data_module, args = _load_data(amr_tokenizer, args)
    
    # model = AMR2TextModelModule(amr_tokenizer, args)
    model = _load_model(load_path, amr_tokenizer, args)
    
    _train(model, data_module, args)

import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_model_specific_args(parser, os.getcwd())
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.gpus=[1]

    main(args)

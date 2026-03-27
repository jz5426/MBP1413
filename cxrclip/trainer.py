import logging
import math
import os
import shutil
from typing import Dict

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch import amp
import nltk

from . import util
from .data import DataModule
from .loss import build_loss
from .model import build_model
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .online_evaluator import OnlineEvaluator

log = logging.getLogger(__name__)

def load_previous_stage_vlm_ckpt(cfg, model):
    if 'vlm_pretrained_ckpt' not in cfg or cfg['vlm_pretrained_ckpt'] is None:
        return model

    if cfg['vlm_pretrained_ckpt'] is not None:
        assert os.path.exists(cfg['vlm_pretrained_ckpt']), 'the vlm checkpoint does not exist.'
        ckpt = torch.load(cfg['vlm_pretrained_ckpt'], map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        log.info("[TRAINER] Successfully loaded the previously trained VLM (image and text) model weights")
        # print('missing: ', missing)
        # print('unexpected: ', unexpected)
    return model

def run(local_rank, cfg: Dict):
    if "tokenizer" in cfg:
        assert (
            cfg["tokenizer"]["pretrained_model_name_or_path"] == cfg["model"]["text_encoder"]["name"]
        ), "tokenizer should be same to text_encoder"
        assert (
            cfg["tokenizer"]["cache_dir"] == cfg["model"]["text_encoder"]["cache_dir"]
        ), "cache directory should be the same if the tokenizer and text_encoder name are the same"
        assert (
            cfg['tokenizer']['dual_cls'] == cfg["model"]["text_encoder"]['dual_cls']
        ), "dual classification tokens should be consistent for both tokenizer and text encoder"

    distributed = local_rank != -1
    if distributed:
        # Initialize the distributed process group
        dist.init_process_group(backend="nccl")
        # Bind the process to a GPU
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        # Wait for all GPUs to join the group before proceeding.
        # If one GPU fails to connect, the others will wait here instead of crashing later.
        dist.barrier() 
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"DistEnv: {util.GlobalEnv.get()}")

    log.info(f"{device}: Load datasets")
    data_config = {}
    if "data_train" in cfg:
        data_config["train"] = cfg["data_train"]
    if "data_valid" in cfg:
        data_config["valid"] = cfg["data_valid"]
    if "data_test" in cfg:
        data_config["test"] = cfg["data_test"]

    # if cfg["model"]["image_encoder"]["name"] == "resnet":
    #     for _split in data_config:
    #         for _dataset in data_config[_split]:
    #             data_config[_split][_dataset]["normalize"] = "imagenet"

    datamodule = DataModule(
        data_config=data_config,
        dataloader_config=cfg["dataloader"],
        tokenizer_config=cfg["tokenizer"] if "tokenizer" in cfg else None,
        loss_config=cfg["loss"],
        transform_config=cfg["transform"]
    )
    train_dataloader, train_sampler = datamodule.train_dataloader(distributed=distributed)
    # valid_dataloaders = datamodule.valid_dataloader(distributed=distributed)
    non_ddp_valid_dataloaders = datamodule.valid_dataloader(distributed=False)
    # non_ddp_valid_dataloaders = datamodule.test_dataloader()
    # test_dataloaders = datamodule.test_dataloader() if len(datamodule.datasets['test']) > 0 else None

    log.info(f"{device}: Build the model")
    model = build_model(cfg["model"], cfg["loss"], config=cfg, tokenizer=datamodule.tokenizer)
    model = load_previous_stage_vlm_ckpt(cfg, model)
    model = model.to(device)
    if distributed:
        # Since DP is slower and less scalable, most people (and PyTorch itself) recommend DDP. 
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    if util.GlobalEnv.get().master:
        log.info(f"{device}: Model info:\n{model}")

    log.info(f"{device}: Build the loss function")
    loss_func = build_loss(cfg["classfication_loss"] if 'classfication_loss' in cfg else cfg["loss"])

    log.info(f"{device}: Build the optimizer")
    optimizer = build_optimizer(model, cfg["optimizer"])

    log.info(f"{device}: Build the LR scheulder")
    if "total_epochs" in cfg["scheduler"]["config"]:
        # with open_dict(cfg):
        cfg["scheduler"]["config"]["total_steps"] = len(train_dataloader) * cfg["scheduler"]["config"]["total_epochs"]
    if "warmup_epochs" in cfg["scheduler"]["config"]:
        # with open_dict(cfg):
        if isinstance(cfg["scheduler"]["config"]["warmup_epochs"], int):
            cfg["scheduler"]["config"]["warmup_steps"] = len(train_dataloader) * cfg["scheduler"]["config"]["warmup_epochs"]
        elif isinstance(cfg["scheduler"]["config"]["warmup_epochs"], float):
            cfg["scheduler"]["config"]["warmup_steps"] = cfg["scheduler"]["config"]["warmup_epochs"]

    scheduler = build_scheduler(optimizer, cfg["scheduler"])
    scaler = amp.GradScaler(device) if cfg["base"]["amp"] else None

    if local_rank < 1:
        log.info("loading nltk module from pre-specified directory")
    nltk.data.path.append('/cluster/projects/mcintoshgroup/CXR-CLIP/nltk_data')
    assert nltk.data.find("tokenizers/punkt")

    # train
    log.info(f"{device}: Train the model")
    if "total_epoch" in cfg["scheduler"]:
        total_epochs = cfg["scheduler"]["total_epoch"]
        cfg["scheduler"]["config"]["total_steps"] = total_epochs * len(train_dataloader)
    else:
        total_epochs = math.ceil(cfg["scheduler"]["config"]["total_steps"] / len(train_dataloader))

    # tensorboard
    util.GlobalEnv.get().summary_writer.train = util.DistSummaryWriter(cfg["base"]["output"]["tensorboard"] + "/train")
    util.GlobalEnv.get().summary_writer.valid = util.DistSummaryWriter(cfg["base"]["output"]["tensorboard"] + "/valid")
    util.GlobalEnv.get().summary_writer.global_step = 0
    util.GlobalEnv.get().summary_writer.train.add_text(
        "hyperparams/config", "\n".join(["\t" + line for line in OmegaConf.to_yaml(cfg).splitlines()]), 0
    )
    if util.GlobalEnv.get().master:
        os.makedirs(cfg["base"]["output"]["checkpoint"], exist_ok=True)

    # training
    # best_loss = ckpt['train_loss'] if cfg.test.resume else 9e9
    # epoch_resume = ckpt['epoch'] if cfg.test.resume else 0
    best_pretrain_loss = 9e9
    best_results = None
    best_finetune_auroc, best_finetune_prauc, patience_threshold, best_finetune_epoch = 0, 0, 30, 0
    current_patience = patience_threshold
    patience_starting_epoch = 10 # assume total finetune epoch is 100
    epoch_resume = 0
    log.info(f'[Trainer] Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    for epoch in range(epoch_resume, total_epochs):
        torch.cuda.empty_cache() # Clear PyTorch cache

        # shuffle differently every epoch so that each gpu always have the same shard sees different data content
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss_dict, val_loss_dict_per_dataset = None, None

        # adjust gamma based on gamma schedule for sogclr loss.
        # for lost_class in loss_func.loss_list:
        #     if hasattr(lost_class, 'loss_func') and hasattr(lost_class.loss_func, 'adjust_gamma'):
        #         log.info('Adjusting gamma for sogclr.')
        #         lost_class.loss_func.adjust_gamma(epoch)

        train_loss_dict = train(
            model,
            device,
            loss_func,
            optimizer,
            scheduler,
            train_dataloader,
            epoch,
            total_epochs,
            scaler,
            cfg["scheduler"]["config"]["total_steps"],
        )

        val_loss_dict_per_dataset = None
        # if 'sogclr' not in cfg['loss'] and util.GlobalEnv.get().master: # run the validation in infoNCE in DDP
        #     val_loss_dict_per_dataset = validation_by_infoNCE(
        #         model, 
        #         device, 
        #         loss_func,
        #         # valid_dataloaders,
        #         non_ddp_valid_dataloaders,
        #         epoch, 
        #         total_epochs, 
        #         local_rank, 
        #         is_amp = cfg["base"]["amp"]
        #     )

        valid_eval_results = None
        if non_ddp_valid_dataloaders is not None and util.GlobalEnv.get().master and cfg.get('base', {}).get('mode', 'train') == 'finetune': # run other validation TASKS in non-DDP manner.
            test_evaluator = OnlineEvaluator(cfg, non_ddp_valid_dataloaders, model, datamodule, rank=util.GlobalEnv.get().world_rank)
            valid_eval_results = test_evaluator.evaluate_classifier_online(data_config["valid"]['base']['name'])

        # # tensorboard for validation results and finetuning
        current_finetune_auroc, current_finetune_prauc, currrent_results = 0, 0, None
        if valid_eval_results is not None and cfg.get('base', {}).get('mode', 'train') == 'finetune':
            for task in valid_eval_results: # zeroshot or retrieval
                for dataset_name, stats in valid_eval_results[task].items():
                    # keep track of the essential evaluation stats for each dataset.
                    if 'AUROC(Avg)' in stats:
                        util.GlobalEnv.get().summary_writer.train.add_scalar(
                            f"{dataset_name}/AUROC(Avg)", stats['AUROC(Avg)'], epoch + 1
                        )
                        current_finetune_auroc = stats['AUROC(Avg)']
                        current_finetune_prauc = stats['PR_AUROC(Avg)']
                        currrent_results = stats
                    if 'PR_AUROC(Avg)' in stats:
                        util.GlobalEnv.get().summary_writer.train.add_scalar(
                            f"{dataset_name}/PR_AUROC(Avg)", stats['PR_AUROC(Avg)'], epoch + 1
                        )
                    if 'Accuracy(Avg)' in stats:
                        util.GlobalEnv.get().summary_writer.train.add_scalar(
                            f"{dataset_name}/Accuracy(Avg)", stats['Accuracy(Avg)'], epoch + 1
                        )
                    if 'F1(Avg)' in stats:
                        util.GlobalEnv.get().summary_writer.train.add_scalar(
                            f"{dataset_name}/F1(Avg)", stats['F1(Avg)'], epoch + 1
                        )
                    if 'Recall@1' in stats:
                        util.GlobalEnv.get().summary_writer.train.add_scalar(
                            f"{dataset_name}/Recall@1", stats['Recall@1'], epoch + 1
                        )
                    if 'Recall@5' in stats:
                        util.GlobalEnv.get().summary_writer.train.add_scalar(
                            f"{dataset_name}/Recall@5", stats['Recall@5'], epoch + 1
                        )
                    if 'Recall@10' in stats:
                        util.GlobalEnv.get().summary_writer.train.add_scalar(
                            f"{dataset_name}/Recall@10", stats['Recall@10'], epoch + 1
                        )

        if train_loss_dict:
            for k, v in train_loss_dict.items():
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"loss_per_epoch/{k}", v, epoch + 1)
            
        if val_loss_dict_per_dataset:

            avg_val_loss_per_loss = {"total": 0.0}
            for loss_key in loss_func.loss_list:
                avg_val_loss_per_loss[loss_key.name] = 0.0

            for data_name, loss_dict in val_loss_dict_per_dataset.items():
                for loss_key, v in loss_dict.items():
                    util.GlobalEnv.get().summary_writer.valid.add_scalar(f"loss_per_epoch/{loss_key}/{data_name}", v, epoch + 1)
                    avg_val_loss_per_loss[loss_key] += v

            for loss_key in avg_val_loss_per_loss:
                avg_val_loss_per_loss[loss_key] /= len(non_ddp_valid_dataloaders)
                util.GlobalEnv.get().summary_writer.valid.add_scalar(f"loss_per_epoch/{loss_key}", avg_val_loss_per_loss[loss_key], epoch + 1)

        # perform auxiliary tasks using the master node.
        if util.GlobalEnv.get().master:
            # checkpoint
            filename = os.path.join(cfg["base"]["output"]["checkpoint"], "model")
            checkpoint = f"{filename}-{epoch+1}.tar"
            model_state_dict = model.state_dict() if local_rank == -1 else model.module.state_dict()

            # NOTE: THIS IS WHAT THE SAVED CHECKPONT LOOKS LIKE.
            if cfg.get('base', {}).get('mode', 'train') == 'train':
                torch.save(
                    {
                        "model": model_state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": cfg,
                        "epoch": epoch + 1,
                        "train_loss": train_loss_dict["total"] if train_loss_dict else -1,
                    },
                    checkpoint,
                )
                log.info(f"Epoch {epoch}, last-model saved")

            # best model
            # if val_loss_dict_per_dataset and avg_val_loss_per_loss[cfg["base"]["loss_best"]] < best_pretrain_loss:
            #     shutil.copyfile(checkpoint, f"{filename}-best.tar")
            #     log.info(f"{filename}-best.tar saved")
            #     best_pretrain_loss = avg_val_loss_per_loss[cfg["base"]["loss_best"]]
            
            # NOTE: save the best checkpoint for the finetuning
            if cfg.get('base', {}).get('mode', 'train') == 'finetune' and current_finetune_auroc > best_finetune_auroc:
                assert 'finetune' in filename, 'must save the finetuned classifier in a seperate model.'
                # torch.save(
                #     {
                #         "model": model_state_dict,
                #         "optimizer": optimizer.state_dict(),
                #         "scheduler": scheduler.state_dict(),
                #         "config": cfg,
                #         "epoch": epoch + 1,
                #         "train_loss": train_loss_dict["total"] if train_loss_dict else -1,
                #     },
                #     f"{filename}-best.tar",
                # )
                # log.info(f"{filename}-best.tar saved for finetuning")
                best_finetune_auroc = current_finetune_auroc
                best_finetune_prauc = current_finetune_prauc
                best_results = currrent_results
                best_finetune_epoch = epoch
                current_patience = patience_threshold
                log.info(f'[Trainer] BEST AUROC SO FAR {best_finetune_auroc} AND BEST PRAUC SO FAR {best_finetune_prauc}, at epoch {epoch}.')

            elif epoch >= patience_starting_epoch and current_finetune_auroc < best_finetune_auroc:
                # only start the patience gracing period after warm up, 
                current_patience -= 1

            if cfg.get('base', {}).get('mode', 'train') == 'finetune' and current_patience == 0:
                log.info(f'[Trainer] No improvement for the past {patience_threshold} epochs. Early stopping.')
                break

        # This prevents fast GPUs from starting Epoch 2 before slow GPUs (Rank 0) are ready.
        if distributed:
            dist.barrier()

    util.GlobalEnv.get().summary_writer.train.close()
    util.GlobalEnv.get().summary_writer.valid.close()
    log.info(f"{device}: Training has been completed")

    if cfg.get('base', {}).get('mode', 'train') == 'finetune':
        log.info(f"Best performance in AUROC {best_finetune_auroc} and in PRAUC {best_finetune_prauc} in epoch {best_finetune_epoch}.")
        log.info(f"Detail results: {best_results}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def train(model, device, loss_func, optimizer, scheduler, dataloader, 
          epoch, total_epochs, scaler, total_step, print_step=30):
    model.train()
    steps_per_epoch = len(dataloader)
    # only log the progress on the logging node.
    if util.GlobalEnv.get().local_rank < 1:
        progress_iter = tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch train]", total=steps_per_epoch)
    else:
        progress_iter = enumerate(dataloader)

    avg_loss_dict = {"total": 0.0}
    for k in loss_func.loss_list:
        avg_loss_dict[k.name] = 0.0

    for idx, batch in progress_iter:
        optimizer.zero_grad(set_to_none=True)

        # mixed precision trainin vs regular training
        if scaler:
            with amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(batch, device)
                loss_dict = loss_func(**outputs, is_train=True, current_step=idx, steps_per_epoch=steps_per_epoch)
            total_loss = loss_dict["total"]
            scaler.scale(total_loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch, device)
            loss_dict = loss_func(**outputs, is_train=True)
            total_loss = loss_dict["total"]
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        util.GlobalEnv.get().summary_writer.global_step = scheduler._step_count

        for k in loss_dict:
            avg_loss_dict[k] += loss_dict[k].item()

        if idx % print_step == 0 and util.GlobalEnv.get().local_rank < 1: # use node with local rank is 0 for logging and training
            for k, lr in enumerate(scheduler.get_last_lr()):
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"hyperparam/lr-{k}", lr, scheduler._step_count)
            util.GlobalEnv.get().summary_writer.train.add_scalar("loss", total_loss.item(), scheduler._step_count)

            # the loss/total is here
            for k in loss_dict:
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"loss/{k}", loss_dict[k].item(), scheduler._step_count)

            progress_iter.set_postfix(
                {
                    "lr": [f"{v:.8f}" for v in scheduler.get_last_lr()],
                    "loss": f"{total_loss.item():.6f}",
                    "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                    "CUDA-Util": f"{torch.cuda.utilization(device)}%",
                }
            )
        if total_step == scheduler._step_count:
            break

    for k in avg_loss_dict:
        avg_loss_dict[k] = avg_loss_dict[k] / len(dataloader)

    return avg_loss_dict


def validation_by_infoNCE(model, device, loss_func, dataloader_dict, epoch, total_epochs, local_rank, is_amp, print_step=10):
    """
    Depends on the dataset, evaluate by retrieval and/or zero-shot
    """
    model.eval()
    loss_dict_per_dataset = dict()
    with torch.no_grad():

        # for each of the validation dataset specified in the .yaml file.
        for data_name, dataloader in dataloader_dict.items():
            if data_name not in {'mimic_cxr'}: # NOTE: add all pretraining datasets here
                continue

            avg_loss_dict = {"total": 0.0}
            for loss_key in loss_func.loss_list:
                avg_loss_dict[loss_key.name] = 0.0

            # only the logging node keep track of the progress.
            if util.GlobalEnv.get().local_rank < 1:
                progress_iter = tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch valid]", total=len(dataloader))
            else:
                progress_iter = enumerate(dataloader)

            for idx, batch in progress_iter:
                if is_amp:
                    with amp.autocast(device_type=device.type):
                        outputs = model(batch, device)
                        loss_dict = loss_func(**outputs, is_train=False)
                else:
                    outputs = model(batch, device)
                    loss_dict = loss_func(**outputs, is_train=False)

                # the assert statement indicates that the original loss is already effective batch size loss, this function has no effect.
                if util.GlobalEnv.get().world_size > 1:
                    for loss_key in loss_dict:
                        # gather the contributed loss from each gpu, but each loss is already effective batch size loss.
                        original_loss = loss_dict[loss_key].clone()
                        dist.all_reduce(loss_dict[loss_key], dist.ReduceOp.SUM)
                        loss_dict[loss_key] = loss_dict[loss_key] / util.GlobalEnv.get().world_size
                        assert torch.allclose(original_loss, loss_dict[loss_key]), f'original loss: {original_loss}; normalized loss: {loss_dict[loss_key]}'

                # accumulate the loss for each batch, in this validation dataset
                for loss_key in loss_dict:
                    avg_loss_dict[loss_key] += loss_dict[loss_key].item()

                # only log it in log node for particular batches.
                if (idx % print_step == 0 or idx == len(dataloader) - 1) and local_rank < 1:
                    progress_iter.set_postfix(
                        {
                            "loss": f'{loss_dict["total"]:.6f}',
                            "CUDA-Mem(%)": torch.cuda.memory_usage(device),
                            "CUDA-Util(%)": torch.cuda.utilization(device),
                        }
                    )

            # get the average loss for the total and the model specifics
            for loss_key in avg_loss_dict:
                avg_loss_dict[loss_key] = avg_loss_dict[loss_key] / len(dataloader)

            loss_dict_per_dataset[data_name] = avg_loss_dict
    return loss_dict_per_dataset

import glob
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from cxrclip import seed_everything
from cxrclip.evaluator import Evaluator
from cxrclip.evaluator_utils import average_and_select_best_across_tasks, decide_best_model_overall
log = logging.getLogger(__name__)

def print_evals(evals, metric="Accuracy(Micro)", best="max"):   
    keys = sorted(list(evals.values())[0].keys())
    st = "| model | " + " | ".join(keys) + "|\n"
    st += "| :---- | " + " | ".join([("-" * (len(k) - 1)) + ":" for k in keys]) + "|\n"
    if best == "max":
        best_score = 0.0
    elif best == "min":
        best_score = 9e9
    else:
        raise ValueError("Unknown value for best, got %s" % best)
    best_ckpt = None
    for c, e in evals.items():
        filename = ".".join(c.split("/")[-1].split(".")[:-1])
        st += f"| {filename} | " + " | ".join([f"{e[k]:.3f}" for k in keys]) + " |\n"
        cur_score = e[metric]
        if best == "max":
            if best_score <= cur_score:
                best_score = cur_score
                best_ckpt = filename
        elif best == "min":
            if best_score >= cur_score:
                best_score = cur_score
                best_ckpt = filename
    st += f"Best {metric}: {best_score:.3f}, from {best_ckpt}\n"
    return st

@hydra.main(version_base=None, config_path="configs", config_name="eval_mbp1413")
def main(cfg: DictConfig):
    print('checkpoint path: ', cfg.test.checkpoint)
    seed_everything(cfg.test.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OmegaConf.resolve(cfg)
    if type(cfg.test.checkpoint).__name__ == "ListConfig":
        ckpt_paths = cfg.test.checkpoint
    elif os.path.isdir(cfg.test.checkpoint):
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.test.checkpoint, "*.tar")))
    else:
        ckpt_paths = sorted(glob.glob(cfg.test.checkpoint))

    cfg_dict = OmegaConf.to_container(cfg)
    evaluator = Evaluator(cfg_dict, ckpt_paths, "test")
    save_path = os.path.dirname(ckpt_paths[0])

    # for each dataset, evaluate each checkpoint.
    per_dataset_eval = []
    for test_dataset_name in evaluator.data_loader_dict.keys():
        # this is the main code for evaluation
        print(test_dataset_name)
        evals = {c: evaluator.evaluate_clip_offline(c, test_dataset_name) for c in ckpt_paths}
        per_dataset_eval.append(evals)

        print("print best score")
        st = ""

        if test_dataset_name in {
            'chestxdet10_gt'
        }:
            st += f"\nzeroshot pointing game - {test_dataset_name}\n"
            zeroshot_pg = {k: {_k: _v for _k, _v in v["zeroshot_pointing_game"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_pg, metric="Pointing_Game(Avg)", best="max")

        if test_dataset_name in {
            "chest14", 
            "chexpert", 
            "physician_padchest207", 
            "chestxdet10",
            }:
            st += f"\nzeroshot binary - {test_dataset_name}\n"
            zeroshot_binary = {k: {_k: _v for _k, _v in v["zeroshot_binary"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_binary, metric="AUROC(Avg)", best="max")

        log.info(cfg.test.checkpoint)
        log.info(st)

        with open(os.path.join(save_path, f"results-{test_dataset_name}.txt"), "w") as outfile:
            outfile.write(st)

    task_list = []

    cls_avg = decide_best_model_overall(per_dataset_eval, 'zeroshot_binary', 'AUROC(Avg)')
    if cls_avg:
        task_list.append(cls_avg)

    pt_avg = decide_best_model_overall(per_dataset_eval, 'zeroshot_pointing_game', 'Pointing_Game(Avg)')
    if pt_avg:
        task_list.append(pt_avg)

    if len(task_list) > 0:
        average_and_select_best_across_tasks(task_list)
    return

if __name__ == "__main__":
    main()

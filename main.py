import logging
import os
import shutil
import subprocess
from pprint import pformat

import torch
import torch.nn

from distributed import comm, launch, get_dist_url
from trainer import SCRLTrainer, ResultPack, TaskState
import utils
from utils import get_cfg, Config, logger, Colorer, Watch, ExceptionLogger
from utils import set_file_handler, set_stream_handler, parse_args

from eimmo_sdk_inhouse import Phase
from eimmo_sdk_inhouse import eimmo

log = logging.getLogger('main')
log_comm = logging.getLogger('comm')
C = Colorer.instance()

id_list = ["6045d6aecbb67023b0e62922",
"604081897b6123519b1e7ec1",
"6007f1751d6abc10d25364dc",
"6007ef768e3ed72df90308c7",
"5ff53f9e279d05bcbf241594",
"5fe86f7f82320da589e77095",
"5fc397e9e731f5bc070de252",
"5fa7715fb61f981d38da27ec",
"5fd9952fcca9e11024cf9860",
"5f905449dd5ee0a18c909a17",
"5f8d83a6c3e03fb94a40676b"]


def download_dataset(id_):
    param = {"prj_id":id_,
            "prj_name":"patrick",
            "prj_type":"bbox",
            "phase":"inhouse_prod",
            "dataset":"/",
            "output":"dataset/"}

    download_image = ["eimmo-inhouse",
            "download-files",
            "-p",
            param["phase"],
            "-pid",
            param["prj_id"],
            "--path",
            param["dataset"],
            "-o",
            param["output"],
            "-k",
            "down_quality"]
    subprocess.run(download_image)


@ExceptionLogger("error")
def main(id_):
    cfg = initialize()
    # cfg.load_dir = id_
    if not cfg.spawn_ctx:
        return False
    
    with Watch('train()'):
        results = []
        for result in train(cfg):
            if result.state == TaskState.TRAIN:
                results.append(result.value['pack'])

    # save metrics throughout the course of the training
    assert result.state == TaskState.DONE
    if results:
        config_super_name = cfg.config_name.split('/')[0]
        result_pack_concat = ResultPack.concat(config_super_name, results)
        result_pack_concat.save_as_csv(cfg.save_dir, 'result_overall')
        result_pack_concat.save_as_plot(cfg.save_dir, config_super_name)
        
    return True


@ExceptionLogger("error")
def train(cfg: Config):
    with Watch('train.run'):
        trainer = SCRLTrainer.init_from_config(cfg=cfg)
        for result in trainer.run():
            yield result


@ExceptionLogger("error")
def initialize(cfg=None):
    cfg = get_cfg(parse_args()) if cfg is None else cfg
    # launch multi-process for DDP
    #   - processes will be branched off at this point
    #   - subprocess ignores launching process and returns None
    if cfg.num_machines * cfg.num_gpus > 1:
        log.info(C.green(f"[!] Lauching Multiprocessing.."))
        cfg.spawn_ctx = launch(main_func=initialize,
                               num_gpus_per_machine=cfg.num_gpus,
                               num_machines=cfg.num_machines,
                               machine_rank=cfg.machine_rank,
                               dist_url=cfg.dist_url,
                               args=(cfg,))  
    else:
        cfg.spawn_ctx = None
    
    # scatter save_dir to all of non-main ranks
    cfg.save_dir = comm.scatter(cfg.save_dir)
    
    # finalize config
    C.set_enabled(not cfg.no_color)  # for sub-processes
    cfg.device = comm.get_local_rank()
    cfg.freeze()
    
    # file logging on the local ranks
    set_stream_handler('comm', cfg.log_level)  # for sub-processes 
    log_rank_file = f"log_rank_{comm.get_rank()}.txt"
    set_file_handler('main', cfg.log_level, cfg.save_dir, log_rank_file)
    set_stream_handler('error', cfg.log_level)
    set_file_handler('error', cfg.log_level, cfg.save_dir, "log_error.txt")
    if comm.is_main_process():
        set_file_handler('result', cfg.log_level, "./", "log_result.txt")

    # log distriubted learning
    if comm.get_world_size() > 1:
        log.info(f"[DDP] dist_url: {cfg.dist_url}")
        log.info(f"[DDP] global_world_size = {comm.get_world_size()}")
        log.info(f"[DDP] num_gpus_per_machine = {torch.cuda.device_count()}")
        log.info(f"[DDP] machine_rank {cfg.machine_rank} / "
                 f"num_machines = {cfg.num_machines}")
        comm.synchronize()
        log_comm.info(f"[DDP] rank (local: {comm.get_local_rank()}, "
                      f"global: {comm.get_rank()}) has been spawned.")
        comm.synchronize()
        log.info(f"[DDP] Synchronized across all the ranks.")
        
    if not cfg.spawn_ctx:
        # This structure (including customized launch.py) is for compatibility 
        # with our internal API. There is no functional difference from the 
        # typical usage of distributed package. Please don't mind this 
        # pecularity and focus on the main algorithm.
        for _ in train(cfg):
            pass
        
    return cfg


if __name__ == '__main__':
    for i in id_list:
        shutil.rmtree('dataset')
        os.rename('runs/scrl_200ep_000',f'runs/{i}')
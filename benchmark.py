import os
from datetime import datetime

import hydra
import polars as pl
import torch
import torch.multiprocessing
import yaml
from hydra.core.config_store import ConfigStore
from tqdm.auto import tqdm

from config import Config
from ts_mamba.checkpoint_manager import load_checkpoint
from ts_mamba.dataloader import get_timeseries_dataloader
from ts_mamba.model import MixerConfig, LinearSequenceModel, EmbeddingSequenceModel


torch.multiprocessing.set_sharing_strategy('file_system')

config_store = ConfigStore.instance()
config_store.store(name="ts_mamba_config", node=Config)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: Config):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device(config.device)
    dtype = torch.bfloat16

    with open(config.benchmark.test_meta, "r") as f:
        test_meta = yaml.safe_load(f)
    context_length = 60 // test_meta["config"]["time_res"] * 24 * config.benchmark.context_window_in_days

    test_loader = get_timeseries_dataloader(
        parquet_path=config.benchmark.test_file,
        meta_path=config.benchmark.test_meta,
        context_length=context_length,
        use_covariates=True,
        batch_size=256,
        num_workers=config.num_workers,
        shuffle=False,
        distributed=False,
    )


    print('==> Building model..')
    d_input = len(test_meta["features"])
    model_config = MixerConfig(
        d_model=config.model.d_model,
        n_layer=config.model.n_layer,
        d_intermediate=config.model.d_intermediate,
        rms_norm=config.model.rms_norm,
        norm_epsilon=config.model.norm_epsilon,
        residual_in_fp32=config.model.residual_in_fp32,
        fused_add_norm=config.model.fused_add_norm,
        ssm_cfg=config.model.ssm_cfg,
        attn_layer_idx=config.model.attn_layer_idx,
        dropout=config.model.dropout,
        attn_cfg=config.model.attn_cfg,
        use_llm_init=config.model.use_llm_init,
        #llm_init_cfg=config.model.llm_init_cfg,
        d_output=config.model.d_output,
        d_input=d_input,
        vocab_size=config.model.vocab_size,
        pad_vocab_multiple=config.model.pad_vocab_multiple,
        tie_embeddings=config.model.tie_embeddings,
    )
    if config.model.architecture == "simple":
        model_cls = LinearSequenceModel
    elif config.model.architecture == "token":
        model_cls = EmbeddingSequenceModel
    else:
        raise ValueError(f"Invalid config.model.architecture {config.model.architecture}")

    dtype = torch.bfloat16
    model = model_cls(cfg=model_config, device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)


    base_dir = hydra.utils.get_original_cwd()
    output_dirname = f"{config.model_tag}_{config.model.architecture}_n{config.model.n_layer}"
    checkpoint_dir = os.path.join(base_dir, "checkpoint", output_dirname) if config.checkpoint_dir is None else config.checkpoint_dir
    model_data, optimizer_data, scheduler_data, meta_data = load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=config.resume_from_step,
        device=device,
        dtype=dtype,
        load_optimizer=True,
        rank=0,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    model = model.to(device)

    print('==> Start benchmark..')
    model.eval()
    preds, tile_ids, ts = [], [], []
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Benchmar")
        for batch_idx, batch in pbar:
            obs, tile_id, target_timestamp = batch["context"].to(device), batch["tile_id"], batch["target_timestamp"]

            raw_preds = model(obs) # (batch, seq, d_output)
            logits = raw_preds[:,-1]
            t = target_timestamp[:,-1]

            preds.append(logits.cpu())
            tile_ids.extend(tile_id[0])
            ts.append(t)

    preds = torch.cat(preds, dim=0)
    ts = torch.cat(ts, dim=0)
    
    records = []
    for (logits, t, tile) in zip(preds, ts, tile_ids):
        records.append([tile, t, logits.tolist()])

    df = pl.DataFrame(
        records,
        schema=["tile_id", "reference_time", "logits"],
        orient="row"
    ).with_columns(
        pl.col("reference_time")
        .cast(pl.Datetime)
        .dt
        .round(f"{test_meta['config']['time_res']}m")
        .alias("reference_time")
    ).with_columns(
        pl.col("reference_time")
        .dt.replace_time_zone("UTC")  # assume timestamps are in UTC
        .dt.convert_time_zone(config.benchmark.time_zone)
        .alias("reference_time_local")
    )

    df.write_parquet(f"benchmark/benchmark_{timestamp_str}.parquet")
    print("Written")

if __name__ == "__main__":
    main()


import torch
from tqdm.auto import tqdm

from ts_mamba.model import RMSELoss

def evaluate_llm(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0.0
    
    # accumulators for metrics
    squared_err_sum = 0.0
    abs_err_sum = 0.0
    total_count = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs = obs.squeeze(-1).to(device)
            targets = targets[:, -1].reshape(-1).to(device)  # (batch,)

            logits = model(obs, num_last_tokens=1).logits     # (batch,1,vocab)
            logits = logits.squeeze(1)                        # (batch,vocab)

            # ----- LOSS -----
            loss = criterion(logits, targets)
            eval_loss += loss.item()

            # ----- EXPECTED VALUE -----
            probs = torch.softmax(logits, dim=-1)             # (batch,vocab)
            vocab_size = probs.size(-1)
            class_vals = torch.arange(vocab_size, device=device, dtype=probs.dtype)
            expected = (probs * class_vals).sum(dim=-1)       # (batch,)

            # ----- ONLINE RMSE + MAE -----
            err = expected - targets
            squared_err_sum += torch.sum(err ** 2).item()
            abs_err_sum += torch.sum(torch.abs(err)).item()
            total_count += targets.numel()

            # Current metrics
            curr_rmse = (squared_err_sum / total_count) ** 0.5
            curr_mae = abs_err_sum / total_count

            pbar.set_description(
                f"Val ({batch_idx+1}/{len(loader)}) | "
                f"Loss: {eval_loss/(batch_idx+1):.3f} | "
                f"RMSE: {curr_rmse:.3f} | "
                f"MAE: {curr_mae:.3f}"
            )

    # Final metrics
    avg_loss = eval_loss / len(loader)
    final_rmse = (squared_err_sum / total_count) ** 0.5
    final_mae = abs_err_sum / total_count

    return {
        "loss": avg_loss,
        "rmse": final_rmse,
        "mae": final_mae,
    }

def evaluate_llm2(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.squeeze(-1).to(device), targets[:,-1].reshape(-1).to(device)
            logits = model(obs, num_last_tokens=1).logits # (batch, 1, vocab_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            eval_loss += loss.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Val loss: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)

    return {
        "loss": avg_eval_loss,
    }

def evaluate_model(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    eval_mae = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            mu, _ = preds
            loss = criterion(preds, targets)
            mae = torch.mean(torch.abs(mu - targets))

            eval_loss += loss.item()
            eval_mae += mae.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Val loss: %.3f | MAE: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1), eval_mae/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)
        avg_eval_mae = eval_mae / len(loader)

    return {
        "loss": avg_eval_loss,
        "mae": avg_eval_mae,
    }

def evaluate_model_rmse(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    eval_rmse = 0
    eval_mae = 0
    eval_zero_mae = 0
    eval_pos_mae = 0

    rmse_crit = RMSELoss()

    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            loss = criterion(preds, targets)
            rmse = rmse_crit(preds, targets)
            mae = torch.mean(torch.abs(preds[:,-1] - targets[:,-1]))

            # Make mask for samples whose last target value == 0
            zero_mask = (targets[:, -1, 0] == 0)
            pos_mask = (targets[:, -1, 0] > 0)

            # Select only the last timestep for preds and targets
            preds_last = preds[:, -1, 0]
            targets_last = targets[:, -1, 0]

            # Safe MAE computation
            if zero_mask.any():
                zero_mae = torch.mean(torch.abs(preds_last[zero_mask] - targets_last[zero_mask]))
            else:
                zero_mae = torch.tensor(0.0, device=preds.device)

            if pos_mask.any():
                pos_mae = torch.mean(torch.abs(preds_last[pos_mask] - targets_last[pos_mask]))
            else:
                pos_mae = torch.tensor(0.0, device=preds.device)

            eval_loss += loss.item()
            eval_rmse += rmse.item()
            eval_mae += mae.item()
            eval_zero_mae += zero_mae.item()
            eval_pos_mae += pos_mae.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Loss: %.3f | RMSE: %.3f | MAE: %.3f | zero-MAE: %.3f | pos-MAE: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1), eval_rmse/(batch_idx+1), eval_mae/(batch_idx+1), eval_zero_mae/(batch_idx+1), eval_pos_mae/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)
        avg_eval_loss_last = eval_rmse / len(loader)
        avg_eval_mae = eval_mae / len(loader)
        avg_eval_zero_mae = eval_zero_mae / len(loader)
        avg_eval_pos_mae = eval_pos_mae / len(loader)

    return {
        "loss": avg_eval_loss,
        "rmse": avg_eval_loss_last,
        "mae": avg_eval_mae,
        "zero_mae": avg_eval_zero_mae,
        "pos_mae": avg_eval_pos_mae,
    }


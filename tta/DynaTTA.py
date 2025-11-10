# Adapted from  https://github.com/kimanki/TAFAS
# Disclaimer: Refactored and documented with the help of an LLM

from typing import List
from collections import deque
from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt

def build_adapter(cfg, model, norm_module=None):
    return DynaTTAAdapter(cfg, model, norm_module)


def setConfig(_C):
    _C.TTA.DYNATTA = CN()
    _C.TTA.DYNATTA.MSE_BUFFER_SIZE= 256
    _C.TTA.DYNATTA.METRIC_HISTORY_SIZE= 256
    _C.TTA.DYNATTA.ALPHA_MIN= 1e-4
    _C.TTA.DYNATTA.ALPHA_MAX= 1e-3
    _C.TTA.DYNATTA.KAPPA= 1.0
    _C.TTA.DYNATTA.ETA= 0.1
    _C.TTA.DYNATTA.EPS= 1e-6
    _C.TTA.DYNATTA.WARMUP_FACTOR= 1
    _C.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL= 1
    _C.TTA.DYNATTA.UPDATE_METRICS_INTERVAL= 1
    _C.TTA.DYNATTA.RTAB_SIZE= 360
    _C.TTA.DYNATTA.RDB_SIZE= 100
    return _C

class DynaTTAAdapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(DynaTTAAdapter, self).__init__()
        self.cfg = cfg
        self.cfg = setConfig(cfg)
        self.model = model
        self.norm_module = norm_module
        self.norm_method = get_norm_method(cfg)

        # TAFAS calibration module (gating)
        cfg.TTA.TAFAS.CALI_MODULE = True
        if cfg.TTA.TAFAS.CALI_MODULE:
            self.cali = Calibration(cfg).to(next(self.model.parameters()).device)

        # freeze everything then unfreeze adapter params
        self._freeze_all_model_params()
        self.named_modules_to_adapt = self._get_named_modules_to_adapt()
        self._unfreeze_modules_to_adapt()
        self.named_params_to_adapt = self._get_named_params_to_adapt()
        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), cfg.TTA)

        # backup state
        self.model_state, self.opt_state = self._copy_state()

        # data loader
        cfg.TEST.BATCH_SIZE = len(get_test_dataloader(cfg).dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        # TAFAS pointers
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_end = {}
        self.inputs_hist = {}
        self.n_adapt = 0

        # output metrics
        self.mse_all = []
        self.mae_all = []

        # DynaTTA buffers & histories
        dyn = cfg.TTA.DYNATTA
        # MSE z-score buffer
        self.mse_buffer = deque(maxlen=dyn.MSE_BUFFER_SIZE)
        # RTAB: sample_id -> [embedding (cuda Tensor), mse, alpha]
        self.rtab = {}
        # RDB: sample_id -> [embedding (cuda Tensor), mse]
        self.rdb = {}
        # metric histories for normalization
        self.metric_hist = [deque(maxlen=dyn.METRIC_HISTORY_SIZE) for _ in range(3)]
        # adaptation-rate and warmup
        self.alpha_t = dyn.ALPHA_MIN      # current adaptation rate
        self.alpha_min = dyn.ALPHA_MIN
        self.alpha_max = dyn.ALPHA_MAX
        self.kappa = dyn.KAPPA            # sensitivity scale
        self.eta = dyn.ETA                # smoothing factor
        self.eps = dyn.EPS                # numerical stability
        self.warmup_steps = dyn.WARMUP_FACTOR * cfg.DATA.PRED_LEN
        self.lr_history = []
        # update intervals
        self.buffer_interval = dyn.UPDATE_BUFFERS_INTERVAL
        self.metric_interval = dyn.UPDATE_METRICS_INTERVAL

        self.steps_since_last_buffer_update = self.steps_since_last_metric_update = 2-2

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _freeze_all_model_params(self):
        for model in self._get_all_models():
            for param in model.parameters():
                param.requires_grad_(False)

    def _get_all_models(self):
        models = [self.model]
        if self.norm_module is not None:
            models.append(self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            models.append(self.cali)
        return models

    def _get_named_modules(self):
        named_modules = []
        for model in self._get_all_models():
            named_modules += list(model.named_modules())
        return named_modules

    def _get_named_modules_to_adapt(self) -> List[str]:
        named_modules = self._get_named_modules()
        if self.cfg.TTA.MODULE_NAMES_TO_ADAPT == 'all':
            return named_modules
        
        named_modules_to_adapt = []
        for module_name in self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(','):
            exact_match = '(exact)' in module_name
            module_name = module_name.replace('(exact)', '')
            if exact_match:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if name == module_name]
            else:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if module_name in name]

        assert len(named_modules_to_adapt) > 0
        return named_modules_to_adapt

    def _get_named_params_to_adapt(self):
        named_params_to_adapt = {}
        for model in self._get_all_models():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    named_params_to_adapt[name] = param
        return named_params_to_adapt

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.opt_state)

    def adapt(self):
        return self.adapt_tafas()

    @torch.enable_grad()
    def adapt_tafas(self):
        self.switch_eval()
        batch_start, batch_end, batch_idx = 0, 0, 0

        for _, inputs in enumerate(self.test_loader):
            enc_all, enc_stamp, dec_all, dec_stamp = prepare_inputs(inputs)
            total = enc_all.shape[0]

            while batch_end < total:
                # determine batch / period
                enc0 = enc_all[batch_start]
                if self.cfg.TTA.TAFAS.PAAS:
                    period, bs = self._calc_period(enc0)
                else:
                    bs = self.cfg.TTA.TAFAS.BATCH_SIZE
                    period = bs - 1
                batch_end = min(batch_start + bs, total)
                bs = batch_end - batch_start
                self.cur_step += bs
                self.steps_since_last_buffer_update += bs
                self.steps_since_last_metric_update += bs

                window = (
                    enc_all[batch_start:batch_end],
                    enc_stamp[batch_start:batch_end],
                    dec_all[batch_start:batch_end],
                    dec_stamp[batch_start:batch_end]
                )
                self.pred_end[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_hist[batch_idx] = window

                # full-ground truth adaptation and buffer updates
                self._adapt_full()
                # partial adaptation
                pred, gt = self._adapt_partial(window, period, bs, batch_idx)

                # optional adjust prediction
                z, dr, dp = self._collect_current_metrics(window)
                device = next(self.model.parameters()).device
                metrics = torch.tensor([z, dr, dp], device=device)
                if self.cfg.TTA.TAFAS.ADJUST_PRED:
                    pred, gt = self._adjust_prediction(pred, window, bs, period, metrics)

                # log metrics
                mse = F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)

                batch_start = batch_end
                batch_idx += 1

        # finalize
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        print('After TSF-TTA of TAFAS')
        print(f'Number of adaptations: {self.n_adapt}')
        print(f'Test MSE: {self.mse_all.mean():.4f}, Test MAE: {self.mae_all.mean():.4f}')
        self.model.eval()
        self.plot_lr_history()
        return self.mse_all.mean(), self.mae_all.mean()

    # ------------------ Core enhancements ------------------
    @torch.no_grad()
    def _adjust_prediction(self, pred, window, batch_size, period, metrics):
        """
        After adaptation, use the gating‐conditioned calibration to re-forecast and splice
        the adapted tail back into the original predictions.
        Args:
            pred (Tensor): original batch predictions, shape [B, H, C]
            window (tuple): (enc_window, enc_stamp, dec_window, dec_stamp)
            batch_size (int): B
            period (int): number of steps used for partial adaptation
            metrics (Tensor): [z, dist_rtab, dist_rdb] for gating
        Returns:
            adjusted_pred (Tensor), ground_truth (Tensor)
        """
        # 1) Input calibration with current shift metrics
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            window_cal = self.cali.input_calibration(window, metrics)
        else:
            window_cal = window

        # 2) Re-forecast on the adapted, calibrated model
        pred_after, gt = forecast(self.cfg, window_cal, self.model, self.norm_module)

        # 3) Output calibration
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            pred_after = self.cali.output_calibration(pred_after, metrics)

        # 4) Splice the adapted tail into the original predictions
        #    For each sample i, replace pred[i, period-i:] with the adapted tail
        for i in range(batch_size - 1):
            pred[i, period - i:] = pred_after[i, period - i:]

        return pred, gt

    @torch.enable_grad()
    def _adapt_full(self):
        # when full GT ready, update buffers, metrics, then adapt
        while self.pred_end and self.cur_step >= self.pred_end[min(self.pred_end)]:
            idx = min(self.pred_end)
            window = self.inputs_hist.pop(idx)
            self.pred_end.pop(idx)

            # 1) Update buffers & adaptation rate (if on schedule)
            device = next(self.model.parameters()).device
            if self.steps_since_last_buffer_update >= self.buffer_interval:
                z, dr, dp = self._compute_and_update_buffers(window)
                self.steps_since_last_buffer_update = 0
                if self.steps_since_last_metric_update >= self.metric_interval:
                    self._update_adaptation_rate(z, dr, dp)
                    self.steps_since_last_metric_update = 0
                # pack metrics tensor once per‐step
                metrics = torch.tensor([z, dr, dp], device=device)
            else:
                metrics = torch.zeros(3, dtype=torch.float32, device=device)


            # 2) Do TAFAS adaptation steps
            for _ in range(self.cfg.TTA.TAFAS.STEPS):
                self.n_adapt += 1
                self.switch_train()

                # pass metrics into calibration
                if hasattr(self, 'cali'):
                    window = self.cali.input_calibration(window, metrics)

                pred, gt = forecast(self.cfg, window, self.model, self.norm_module)

                if hasattr(self, 'cali'):
                    pred = self.cali.output_calibration(pred, metrics)

                loss = F.mse_loss(pred, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.switch_eval()

    @torch.enable_grad()
    def _adapt_partial(self, window, period, batch_size, batch_idx):
        # 1) Partial‐MSE RTAB update (if on buffer schedule)
        if self.steps_since_last_buffer_update >= self.buffer_interval:
            self._update_rtab_partial(window, period, batch_idx)

        # 2) Recompute metrics & adaptation rate (if on metric schedule)
        device = next(self.model.parameters()).device
        if self.steps_since_last_metric_update >= self.metric_interval:
            z, dr, dp = self._collect_current_metrics(window)
            self._update_adaptation_rate(z, dr, dp)
            metrics = torch.tensor([z, dr, dp], device=device)
            self.steps_since_last_metric_update = 0
        else:
            metrics = torch.zeros(3, dtype=torch.float32, device=device)

        # 3) Partial‐GT adaptation steps
        for _ in range(self.cfg.TTA.TAFAS.STEPS):
            self.n_adapt += 1

            if hasattr(self, 'cali'):
                window = self.cali.input_calibration(window, metrics)

            pred, gt = forecast(self.cfg, window, self.model, self.norm_module)

            if hasattr(self, 'cali'):
                pred = self.cali.output_calibration(pred, metrics)

            pred_p, gt_p = pred[0][:period], gt[0][:period]
            loss = F.mse_loss(pred_p, gt_p)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return pred, gt

    def _collect_current_metrics(self, window):
        """
        Recompute the shift metrics using the most recent full-window MSE in the buffer
        plus embedding distances, handling the case where the buffer is not yet populated.
        """
        # 1) Z-score of the latest full-window MSE (or 0 if buffer is empty)
        if len(self.mse_buffer) == 0:
            z = 0.0
        else:
            last_mse = self.mse_buffer[-1]
            mu    = np.mean(self.mse_buffer)
            sigma = np.std(self.mse_buffer)
            z     = (last_mse - mu) / (sigma + self.eps)

        # 2) Embedding distances (will return 0.0 if RTAB/RDB empty)
        dr = self._dist_rtab(window)
        dp = self._dist_rdb(window)

        return z, dr, dp


    def _compute_and_update_buffers(self, window):
        """
        Forecasts the full window with the frozen model, then updates:
          - MSE buffer & computes per-sample z-scores
          - RTAB (per sample) and RDB
          - Returns a single aggregated z, and current RTAB/RDB distances
        """
        # 1) Forward pass with frozen model
        pred, gt = forecast(self.cfg, window, self.model, self.norm_module)

        # 2) Compute per-sample full-window MSEs: shape [B]
        mse_per_sample = F.mse_loss(pred, gt, reduction='none')          \
            .mean(dim=(-2, -1))                                             # [B]
        mse_per_sample = mse_per_sample.detach().cpu().numpy()            # to numpy for buffer

        # 3) Update MSE buffer & compute z-scores per sample
        z_list = []
        for mse in mse_per_sample:
            self.mse_buffer.append(mse)
            mu = np.mean(self.mse_buffer)
            sigma = np.std(self.mse_buffer)
            z_list.append((mse - mu) / (sigma + self.eps))
        # Aggregate z-scores (e.g., mean)
        z = float(np.mean(z_list))

        # 4) Update RTAB and RDB for each sample in the batch
        batch_size = len(mse_per_sample)
        # Compute sample IDs for this batch: assume samples correspond to times [t-batch_size+1 ... t]
        for i, mse in enumerate(mse_per_sample):
            sid = self.cur_step - batch_size + 1 + i
            # extract embedding for the i-th sample only
            emb = self._extract_embedding((
                window[0][i:i+1], window[1][i:i+1],
                window[2][i:i+1], window[3][i:i+1]
            )).detach().cpu()
            # full MSE has alpha=1.0
            self._update_rtab_full(sid, emb, float(mse))

        # 5) Compute embedding distances
        dr = self._dist_rtab(window)
        dp = self._dist_rdb(window)

        return z, dr, dp

    def _update_rtab_full(self, sid, emb, mse_full):
        """
        Inserts or updates a single RTAB entry for sample `sid` with full MSE and alpha=1.0,
        then enforces RTAB capacity and updates RDB accordingly.
        """
        # Store embedding, full MSE, confidence alpha=1.0
        self.rtab[sid] = [emb, mse_full, 1.0]
        # Enforce RTAB capacity: remove oldest if needed
        if len(self.rtab) > self.cfg.TTA.DYNATTA.RTAB_SIZE:
            oldest = min(self.rtab.keys())
            del self.rtab[oldest]
        # Update RDB using this new full-MSE entry
        self._update_rdb(sid, emb, mse_full)

    def _update_rtab_partial(self, window, l, idx):
        # for each sample in batch, update partial MSE and alpha
        for b in range(window[0].shape[0]):
            sid = self.cur_step - window[0].shape[0] + b
            emb = self._extract_embedding((
                window[0][b:b+1], window[1][b:b+1], window[2][b:b+1], window[3][b:b+1]
            )).detach().cpu()
            single_win = (
                window[0][b:b+1], window[1][b:b+1],
                window[2][b:b+1], window[3][b:b+1]
            )
            pred_b, gt_b = forecast(self.cfg, single_win, self.model, self.norm_module)
            pred_p = pred_b[0, :l]   # shape [l, C]
            gt_p   = gt_b[0, :l]     # shape [l, C]
            mse_p = F.mse_loss(pred_p, gt_p).item()
            alpha = l / self.cfg.DATA.PRED_LEN
            self.rtab[sid] = [emb, mse_p, alpha]
            if len(self.rtab) > self.cfg.TTA.DYNATTA.RTAB_SIZE:
                oldest = min(self.rtab)
                del self.rtab[oldest]

    def _update_rdb(self, sid, emb, mse):
        cap = self.cfg.TTA.DYNATTA.RDB_SIZE
        if sid in self.rdb:
            if mse < self.rdb[sid][1]:
                self.rdb[sid] = [emb, mse]
        else:
            if len(self.rdb) < cap:
                self.rdb[sid] = [emb, mse]
            else:
                # replace worst
                worst = max(self.rdb.items(), key=lambda x: x[1][1])[0]
                if mse < self.rdb[worst][1]:
                    del self.rdb[worst]
                    self.rdb[sid] = [emb, mse]

    def _dist_rtab(self, window):
        # L2 between current embedding and weighted avg of rtab
        if not self.rtab: return 0.0
        embs, mses, alps = zip(*self.rtab.values())
        inv = np.array([alp / (m + self.eps) for m, alp in zip(mses, alps)], dtype=float)
        w = inv / inv.sum()
        device = next(self.model.parameters()).device
        stack = torch.stack(embs, 0).to(device)
        w_tensor = torch.from_numpy(w).to(device).view(-1, 1, 1, 1)
        avg = (stack * w_tensor).sum(0)
        cur = self._extract_embedding(window).to(device)
        return torch.norm(cur - avg, p=2, dim=-1).mean().item()

    def _dist_rdb(self, window):
        if not self.rdb: return 0.0
        embs, mses = zip(*self.rdb.values())
        inv = np.array([1.0 / (m + self.eps) for m in mses], dtype=float)
        w = inv / inv.sum()
        device = next(self.model.parameters()).device
        stack = torch.stack(embs, 0).to(device)
        avg = (stack * torch.from_numpy(w).to(device).view(-1, 1, 1, 1)).sum(0)
        cur = self._extract_embedding(window).to(device)
        return torch.norm(cur - avg, p=2, dim=-1).mean().item()

    # ---------- metrics & adaptation rate ----------
    def _update_adaptation_rate(self, z, dr, dp):
        # normalize each metric
        print("in update_adaptation_rate")
        norms = []
        for i, m in enumerate([z, dr, dp]):
            hist = self.metric_hist[i]
            hist.append(m)
            mu, sd = np.mean(hist), np.std(hist)
            norms.append((m - mu) / (sd + self.eps))
        S = sum(norms)
        lam = 1 + (self.alpha_max / self.alpha_min - 1) / (1 + math.exp(-self.kappa * S))
        # warm-up
        gamma = min(1.0, self.n_adapt / (self.warmup_steps + self.eps))
        alpha_tgt = self.alpha_min * (1 + gamma * (lam - 1))
        # smooth
        self.alpha_t += self.eta * (alpha_tgt - self.alpha_t)
        # set lr
        for g in self.optimizer.param_groups:
            g['lr'] = float(self.alpha_t)
        self.lr_history.append(float(self.alpha_t))

    # ---------- utilities ----------
    def _copy_state(self):
        return deepcopy(self.model.state_dict()), deepcopy(self.optimizer.state_dict())

    def switch_train(self):
        self.model.train()
        if hasattr(self, 'cali'): self.cali.train()

    def switch_eval(self):
        self.model.eval()
        if hasattr(self, 'cali'): self.cali.eval()

    def _freeze_all(self):
        for p in list(self.model.parameters()) + (list(self.norm_module.parameters()) if self.norm_module else []) + (
                list(self.cali.parameters()) if hasattr(self, 'cali') else []):
            p.requires_grad_(False)

    def _find_modules_to_adapt(self):
        mods = list(self.model.named_modules()) + (list(self.norm_module.named_modules()) if self.norm_module else []) + (
               list(self.cali.named_modules()) if hasattr(self, 'cali') else [])
        if self.cfg.TTA.MODULE_NAMES_TO_ADAPT == 'all':
            return mods
        chosen = []
        for name in self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(','):
            exact = '(exact)' in name
            key = name.replace('(exact)', '')
            if exact:
                chosen += [(n,m) for n,m in mods if n == key]
            else:
                chosen += [(n,m) for n,m in mods if key in n]
        return chosen

    def _unfreeze_modules_to_adapt(self):
        for _, module in self.named_modules_to_adapt:
            module.requires_grad_(True)

    def _find_params_to_adapt(self):
        return {n:p for n,p in list(self.model.named_parameters()) + (list(self.norm_module.named_parameters()) if self.norm_module else []) + (
                   list(self.cali.named_parameters()) if hasattr(self, 'cali') else []) if p.requires_grad}

    def _calc_period(self, enc0):
        fft = torch.fft.rfft(enc0 - enc0.mean(0), dim=0)
        amp = fft.abs(); pw = amp.pow(2).mean(0)
        try:
            per = enc0.shape[0] // fft[:, pw.argmax()].argmax().item()
        except:
            per = 24
        per *= self.cfg.TTA.TAFAS.PERIOD_N
        return per, per + 1

    def _extract_embedding(self, window):
        x_enc, x_mark_enc, _, _ = prepare_inputs(window)
        emb = self.model.enc_embedding(x_enc, x_mark_enc)
        out, _ = self.model.encoder(emb, attn_mask=None)
        return out

    def plot_lr_history(self):
        fig, ax = plt.subplots()
        print(self.lr_history)
        ax.plot(self.lr_history, alpha=0.9, color="#9238B4")

        ax.set_xlabel(r'Step', fontsize=14, labelpad=10)
        ax.set_ylabel(r'Adaptation Rate', fontsize=14, labelpad=10)
        ax.set_title(f'Adaptation rate vs Step', fontsize=14, pad=10)

        ax.set_facecolor('#EAEAF2')
        ax.legend(facecolor="#EAEAF2", prop={'size': 8}, loc='upper right')

        ax.tick_params(axis=u'both', which=u'both',length=0, pad=9.5, labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.ticklabel_format(useOffset=False)

        ax.grid(True, color='white')
        fig.tight_layout()
        plt.savefig(f"/content/drive/My Drive/DynaTTA/plots/plot_lr_history_DYNATTA_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_{self.cfg.DATA.PRED_LEN}_warmup{self.cfg.TTA.DYNATTA.WARMUP_FACTOR}_buffer_update_step_{self.cfg.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL}.pdf",bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f"/content/drive/My Drive/DynaTTA/plots/plot_lr_history_DYNATTA_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_{self.cfg.DATA.PRED_LEN}_warmup{self.cfg.TTA.DYNATTA.WARMUP_FACTOR}_buffer_update_step_{self.cfg.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL}.png",bbox_inches='tight', pad_inches=0.0)

        import pandas as pd
        pd.DataFrame(self.lr_history, columns=["lr"]).to_csv(
            f"/content/drive/My Drive/DynaTTA/plots/lr_history_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_{self.cfg.DATA.PRED_LEN}_warmup{self.cfg.TTA.DYNATTA.WARMUP_FACTOR}_buffer_update_step_{self.cfg.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL}.csv", 
            index_label="step"
        )
        print(len(self.lr_history))


# --------------- Calibration (Dynamic GCM) ---------------
class DynamicGCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden=64, gating_init=0.01, var_wise=True, metric_dim=3):
        super().__init__()
        self.var_wise = var_wise
        if var_wise:
            self.weight = nn.Parameter(torch.zeros(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.zeros(window_len, window_len))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.static_g = nn.Parameter(gating_init * torch.ones(n_var))
        self.mlp = nn.Sequential(
            nn.Linear(metric_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_var)
        )

    def forward(self, x, metrics):
        # x: [B, L, C]
        if metrics.dim() == 1:
            m = metrics.unsqueeze(0).float()
        else:
            m = metrics
        adj = self.mlp(m).mean(0)
        g = torch.tanh(self.static_g + adj).view(1,1,-1)
        if self.var_wise:
            cal = x + g * (torch.einsum('biv,iov->bov', x, self.weight) + self.bias)
        else:
            cal = x + g * (torch.einsum('biv,io->bov', x, self.weight) + self.bias)
        return cal

class Calibration(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        seq, pred, n_var = cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN, cfg.DATA.N_VAR
        hd, init, vw = cfg.TTA.TAFAS.HIDDEN_DIM, cfg.TTA.TAFAS.GATING_INIT, cfg.TTA.TAFAS.GCM_VAR_WISE
        dim = 3
        if cfg.MODEL.NAME == 'PatchTST':
            self.in_cali = DynamicGCM(seq, 1, hd, init, vw, dim)
            self.out_cali = DynamicGCM(pred, 1, hd, init, vw, dim)
        else:
            self.in_cali = DynamicGCM(seq, n_var, hd, init, vw, dim)
            self.out_cali = DynamicGCM(pred, n_var, hd, init, vw, dim)

    def input_calibration(self, window, metrics=None):
        x_enc, x_mark, x_dec, d_mark = prepare_inputs(window)
        return (self.in_cali(x_enc, metrics), x_mark, x_dec, d_mark)

    def output_calibration(self, out, metrics=None):
        return self.out_cali(out, metrics)
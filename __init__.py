from __future__ import annotations
import math
import types
import hashlib
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import comfy.utils
import comfy.patcher_extension

try:
    import latent_preview
except Exception:
    latent_preview = None

import comfy.ldm.flux.math
import comfy.ldm.flux.layers
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention

_PREFIX = '[UntwistingRoPE-Klein9b]'

_RF_LAST_DEBUG_STORE: Dict[str, Any] = {
    'cache': {},
    'sampler_sigmas': None,
    'built_sigmas': None,
    'run_count': 0,
}

_RF_PERSISTENT_TRAJECTORY_CACHE: Dict[str, Dict[str, Any]] = {}
_RF_PERSISTENT_CACHE_MAX_ITEMS = 4

# ═══════════════════════════════════════════════════════════════════════════════
# Native Attention Interceptor (Double-bound to math and layers namespaces)
# ═══════════════════════════════════════════════════════════════════════════════

def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)

def _adain(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t_mean = target.mean(dim=2, keepdim=True)
    s_mean = style.mean(dim=2, keepdim=True)
    t_std  = target.float().var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std  = style.float().var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean

def _cross_batch_adain_qk_generic(
    xq: torch.Tensor, xk: torch.Tensor, 
    xq_r: torch.Tensor, xk_r: torch.Tensor, 
    s: int, e: int, strength: float, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = max(0.0, min(1.0, strength))
    if a <= 0.0 or e <= s:
        return xq, xk
    
    q_t, k_t = xq[:, :, s:e], xk[:, :, s:e]
    q_r, k_r = xq_r[:, :, s:e], xk_r[:, :, s:e]
    
    xq[:, :, s:e] = q_t * (1.0 - a) + _adain(q_t, q_r, eps) * a
    xk[:, :, s:e] = k_t * (1.0 - a) + _adain(k_t, k_r, eps) * a
    return xq, xk

def untwisting_rope_attention_handler(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    pe: Optional[torch.Tensor], 
    mask: Optional[torch.Tensor] = None, 
    transformer_options: Dict[str, Any] = {}
) -> torch.Tensor:
    cfg = transformer_options.get('untwisting_rope_zimage')
    
    if not cfg or not cfg.get('enabled'):
        return _orig_flux_attention(q, k, v, pe, mask, transformer_options)
        
    block_idx = int(transformer_options.get('block_index', -1))
    if not (int(cfg['start_block']) <= block_idx <= int(cfg['end_block'])):
        return _orig_flux_attention(q, k, v, pe, mask, transformer_options)

    # Apply RoPE calculations
    if pe is not None:
        q, k = apply_rope(q, k, pe)
        pe = None

    target_bsz = int(cfg.get('cross_batch_target_batch', 0))
    bsz = q.shape[0]
    heads = q.shape[1]
    
    img_slice = transformer_options.get("img_slice")
    txt_len = img_slice[0] if img_slice else 0
    seq_len = q.shape[2]

    progress   = float(cfg.get('progress', 0.0))
    high_scale = _lerp(cfg['high_scale_start'], cfg['high_scale_end'], progress)
    low_scale  = _lerp(cfg['low_scale_start'],  cfg['low_scale_end'],  progress)
    beta       = float(cfg.get('beta', 2.0))
    head_dim   = q.shape[3]
    
    axes_dims = cfg.get('axes_dims', [16, 56, 56])
    scale_vec = _build_frequency_scale_vector(
        head_dim, axes_dims,
        high_scale, low_scale, beta,
        k.device, k.dtype,
    ).view(1, 1, 1, head_dim)

    # ═══════════════════════════════════════════════════════════════════════════
    # Target-Only Standalone Patching Mode (If no RF Inversion is connected)
    # ═══════════════════════════════════════════════════════════════════════════
    if target_bsz <= 0 or bsz < target_bsz * 2:
        # Scale the keys of the target frame directly
        ref_k_img = k[:, :, txt_len:seq_len] * scale_vec
        k_final = torch.cat([k[:, :, :txt_len], ref_k_img], dim=2)
        return optimized_attention(q, k_final, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options)

    # ═══════════════════════════════════════════════════════════════════════════
    # Reference-Guided Guidance Mode (With RF Inversion connected)
    # ═══════════════════════════════════════════════════════════════════════════
    q_t, k_t, v_t = q[:target_bsz], k[:target_bsz], v[:target_bsz]
    q_r, k_r, v_r = q[target_bsz:target_bsz*2], k[target_bsz:target_bsz*2], v[target_bsz:target_bsz*2]

    ref_k_img = k_r[:, :, txt_len:seq_len] * scale_vec
    ref_v_img = v_r[:, :, txt_len:seq_len]

    if cfg.get('apply_adain') and float(cfg.get('adain_strength', 0)) > 0:
        q_t, k_t = _cross_batch_adain_qk_generic(
            q_t, k_t, q_r, k_r,
            txt_len, seq_len, float(cfg['adain_strength'])
        )

    k_t_final = torch.cat([k_t, ref_k_img], dim=2)
    v_t_final = torch.cat([v_t, ref_v_img], dim=2)

    mask_t = None
    if mask is not None:
        mask_t = mask[:target_bsz]
        ref_len = ref_k_img.shape[2]
        if mask_t.ndim >= 2:
            padding = torch.zeros(
                (*mask_t.shape[:-1], ref_len),
                device=mask_t.device, dtype=mask_t.dtype,
            )
            mask_t = torch.cat([mask_t, padding], dim=-1)

    out_t = optimized_attention(
        q_t, k_t_final, v_t_final,
        heads, skip_reshape=True, mask=mask_t,
        transformer_options=transformer_options
    )

    mask_r = mask[target_bsz:target_bsz*2] if mask is not None and mask.shape[0] >= target_bsz * 2 else None
    out_r = optimized_attention(
        q_r, k_r, v_r,
        heads, skip_reshape=True, mask=mask_r,
        transformer_options=transformer_options
    )

    outs = [out_t, out_r]
    
    if bsz > target_bsz * 2:
        q_extra, k_extra, v_extra = q[target_bsz*2:], k[target_bsz*2:], v[target_bsz*2:]
        mask_extra = mask[target_bsz*2:] if mask is not None else None
        out_extra = optimized_attention(
            q_extra, k_extra, v_extra,
            heads, skip_reshape=True, mask=mask_extra,
            transformer_options=transformer_options
        )
        outs.append(out_extra)

    return torch.cat(outs, dim=0)

# Double-bind the monkey patch to math.py and layers.py namespaces
if not hasattr(comfy.ldm.flux.math, "_untwist_orig_attention"):
    comfy.ldm.flux.math._untwist_orig_attention = comfy.ldm.flux.math.attention
    
comfy.ldm.flux.math.attention = untwisting_rope_attention_handler
comfy.ldm.flux.layers.attention = untwisting_rope_attention_handler

_orig_flux_attention = comfy.ldm.flux.math._untwist_orig_attention
print(f"{_PREFIX} Custom attention successfully double-bound to math.py and layers.py")

# ═══════════════════════════════════════════════════════════════════════════════
# Hashing and Cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_update_tensor(h: "hashlib._Hash", t: torch.Tensor, full: bool = True) -> None:
    td = t.detach().to(device='cpu').contiguous()
    h.update(str(tuple(td.shape)).encode('utf-8'))
    h.update(str(td.dtype).encode('utf-8'))
    if full:
        h.update(td.numpy().tobytes())
    else:
        flat = td.flatten()
        if flat.numel() > 4096:
            flat = flat[torch.linspace(0, flat.numel() - 1, 4096).long()]
        h.update(flat.numpy().tobytes())

def _hash_any(obj: Any, h: Optional["hashlib._Hash"] = None, depth: int = 0) -> str:
    if h is None:
        h = hashlib.sha1()
    if depth > 12:
        h.update(b'<maxdepth>')
        return h.hexdigest()
    if torch.is_tensor(obj):
        h.update(b'TENSOR')
        _hash_update_tensor(h, obj, full=True)
    elif isinstance(obj, dict):
        h.update(b'DICT')
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            if str(k) == 'transformer_options':
                continue
            h.update(str(k).encode('utf-8'))
            _hash_any(obj[k], h, depth + 1)
    elif isinstance(obj, (list, tuple)):
        h.update(b'LIST' if isinstance(obj, list) else b'TUPLE')
        h.update(str(len(obj)).encode('utf-8'))
        for v in obj:
            _hash_any(v, h, depth + 1)
    elif obj is None:
        h.update(b'NONE')
    else:
        h.update(repr(obj).encode('utf-8', errors='ignore'))
    return h.hexdigest()

def _make_rf_persistent_key(
    ref_clean: torch.Tensor,
    ref_conditioning: Any,
    sampler_sigmas: List[float],
    target_b: int,
    rf_mode: str,
    gamma: float,
    gamma_curve: float,
    norm_strength: float,
    cond_mode: str,
    pmi_alpha: float = 0.4,
) -> str:
    h = hashlib.sha1()
    h.update(str(tuple(ref_clean.shape)).encode('utf-8'))
    _hash_update_tensor(h, ref_clean, full=True)
    h.update(_hash_any(ref_conditioning).encode('utf-8'))
    h.update(str([round(float(s), 8) for s in sampler_sigmas]).encode('utf-8'))
    h.update(str(int(target_b)).encode('utf-8'))
    h.update(str(rf_mode).encode('utf-8'))
    h.update(f'{float(gamma):.8f}'.encode('utf-8'))
    h.update(f'{float(gamma_curve):.8f}'.encode('utf-8'))
    h.update(f'{float(norm_strength):.8f}'.encode('utf-8'))
    h.update(str(cond_mode).encode('utf-8'))
    h.update(f'{float(pmi_alpha):.8f}'.encode('utf-8'))
    return h.hexdigest()

def _cache_to_cpu(cache: Dict[float, torch.Tensor]) -> Dict[float, torch.Tensor]:
    return {
        float(k): v.detach().to(device='cpu').clone()
        for k, v in cache.items()
        if torch.is_tensor(v)
    }

def _cache_to_device(cache: Dict[float, torch.Tensor], device, dtype) -> Dict[float, torch.Tensor]:
    return {
        float(k): v.to(device=device, dtype=dtype).detach().clone()
        for k, v in cache.items()
        if torch.is_tensor(v)
    }

def _put_persistent_rf_cache(key: str, entry: Dict[str, Any]) -> None:
    _RF_PERSISTENT_TRAJECTORY_CACHE[key] = entry
    while len(_RF_PERSISTENT_TRAJECTORY_CACHE) > _RF_PERSISTENT_CACHE_MAX_ITEMS:
        oldest = next(iter(_RF_PERSISTENT_TRAJECTORY_CACHE.keys()))
        _RF_PERSISTENT_TRAJECTORY_CACHE.pop(oldest, None)

def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on', 'y', 't')
    return bool(value)

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime stats
# ═══════════════════════════════════════════════════════════════════════════════

class _RuntimeStats:
    def __init__(self, verbose: bool = False, rf_verbose: bool = False) -> None:
        self.verbose: bool = _coerce_bool(verbose)
        self.rf_verbose: bool = _coerce_bool(rf_verbose)
        self.wrapper_calls:  int = 0
        self.patchify_calls: int = 0
        self.attn_calls:     int = 0
        self.context_refiner_calls: int = 0
        self.rf_sigma_cache: Dict[float, torch.Tensor] = {}
        self.rf_eps: Optional[torch.Tensor] = None
        self.rf_prev_z: Optional[torch.Tensor] = None
        self.rf_prev_sigma: Optional[float] = None
        self.rf_step_count: int = 0
        self.rf_run_count: int = 0
        self.rf_sampler_sigmas: Optional[List[float]] = None
        self.rf_schedule_built: bool = False
        self.fixed_noise: Optional[torch.Tensor] = None
        self.scale_vec_logged:  bool = False
        self.joint_mask_logged: bool = False
        self.parameterization: str = 'unknown'

def _vprint(stats: Optional[_RuntimeStats], *args, **kwargs) -> None:
    if stats is not None and _coerce_bool(getattr(stats, 'verbose', False)):
        print(*args, **kwargs)

def _rf_vprint(stats: Optional[_RuntimeStats], *args, **kwargs) -> None:
    if stats is not None and _coerce_bool(getattr(stats, 'rf_verbose', False)):
        print(*args, **kwargs)

def _rf_step_iterator(num_steps: int):
    return range(max(0, int(num_steps)))

def _rf_format_duration(seconds: float) -> str:
    seconds_i = int(max(0, round(float(seconds))))
    minutes, seconds_i = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{seconds_i:02d}'
    return f'{minutes:02d}:{seconds_i:02d}'

def _rf_progress_snapshot(step_i: int, total_steps: int, start_time: float, persistent: bool = False) -> None:
    total_steps = max(1, int(total_steps))
    step_i = max(0, min(int(step_i), total_steps))
    elapsed = max(0.0, time.time() - float(start_time))
    frac = step_i / total_steps
    bar_width = 70
    filled = int(round(bar_width * frac))
    bar = '█' * filled + ' ' * (bar_width - filled)
    percent = int(round(frac * 100.0))
    rate = step_i / elapsed if elapsed > 1e-9 else 0.0
    remaining = max(0.0, (total_steps - step_i) / rate) if step_i > 0 and rate > 1e-9 else 0.0
    rate_text = f'{rate:.2f}it/s' if rate >= 1.0 else f'{(1.0 / max(rate, 1e-9)):.2f}s/it'
    line = (
        f'RF inversion: {percent:3d}%|{bar}| '
        f'{step_i}/{total_steps} '
        f'[{_rf_format_duration(elapsed)}<{_rf_format_duration(remaining)}, {rate_text}]'
    )
    end = '\n' if persistent or step_i >= total_steps else '\r'
    print(line, end=end, flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Parameterization auto-detection
# ═══════════════════════════════════════════════════════════════════════════════

def _velocity_from_pred(
    x_sigma: torch.Tensor,
    pred: torch.Tensor,
    sigma: float,
    parameterization: str,
) -> torch.Tensor:
    if parameterization == 'x0':
        return (x_sigma - pred) / max(float(sigma), 1e-7)
    return pred

# ═══════════════════════════════════════════════════════════════════════════════
# RF utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

_GAMMA_RF_MODES = {'rf_gamma', 'rf_gamma_rk2'}

def _coerce_gamma_curve(value: Any = 0.0) -> float:
    try:
        curve = float(value)
    except Exception:
        curve = 0.0
    if not math.isfinite(curve):
        curve = 0.0
    return max(0.0, min(8.0, curve))

def _normalize_rf_mode_and_gamma_curve(
    mode: str,
    gamma_curve: float = 0.0,
) -> Tuple[str, float]:
    mode = str(mode or 'rf_gamma')
    return mode, _coerce_gamma_curve(gamma_curve)

def _coerce_norm_strength(norm_strength: float) -> float:
    try:
        strength = float(norm_strength)
    except Exception:
        strength = 0.0
    if not math.isfinite(strength):
        strength = 0.0
    return max(0.0, min(1.0, strength))

def _rf_gamma_for_mode(
    mode: str,
    gamma: float,
    sigma_prev: float,
    sigma_cur: float,
    gamma_curve: float = 0.0,
) -> float:
    mode, gamma_curve = _normalize_rf_mode_and_gamma_curve(mode, gamma_curve)
    if mode in ('linear', 'fireflow'):
        return 0.0
    if gamma_curve > 0.0 and mode in _GAMMA_RF_MODES:
        s = max(0.0, min(1.0, 0.5 * (float(sigma_prev) + float(sigma_cur))))
        bell = max(0.0, min(1.0, 4.0 * s * (1.0 - s)))
        return float(gamma) * (bell ** gamma_curve)
    return float(gamma)

def _rf_linear_target(ref_clean: torch.Tensor, eps: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(0.0, min(1.0, float(sigma)))
    return (1.0 - sigma) * ref_clean + sigma * eps

def _rf_match_mean_std(x: torch.Tensor, target: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return x
    dims = tuple(range(1, x.ndim))
    x_mean = x.mean(dim=dims, keepdim=True)
    x_std = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
    t_mean = target.mean(dim=dims, keepdim=True)
    t_std = target.std(dim=dims, keepdim=True).clamp_min(1e-6)
    matched = (x - x_mean) / x_std * t_std + t_mean
    return (1.0 - strength) * x + strength * matched

# ═══════════════════════════════════════════════════════════════════════════════
# PMI
# ═══════════════════════════════════════════════════════════════════════════════

class _PMIState:
    def __init__(self) -> None:
        self.v_mean: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.v_norm_sq_mean: float = 0.0

    def reset(self) -> None:
        self.v_mean = None
        self.step_count = 0

    def update_and_correct(
        self,
        v_model: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        alpha = max(0.0, min(1.0, float(alpha)))
        k = self.step_count

        if self.v_mean is None:
            self.v_mean = v_model.detach().clone()
            self.v_norm_sq_mean = float(v_model.detach().float().pow(2).mean().item())
            self.step_count = 1
            return v_model

        k_new = k + 1
        self.v_mean = (self.v_mean * (k / k_new) + v_model.detach() * (1.0 / k_new)).to(
            device=v_model.device, dtype=v_model.dtype
        )
        self.v_norm_sq_mean = (
            self.v_norm_sq_mean * (k / k_new)
            + float(v_model.detach().float().pow(2).mean().item()) * (1.0 / k_new)
        )
        self.step_count = k_new

        v_corrected = (1.0 - alpha) * v_model + alpha * self.v_mean
        delta_model = v_model - self.v_mean
        delta_corr  = v_corrected - self.v_mean

        r_sq = float(delta_model.detach().float().pow(2).mean().item())
        c_sq = float(delta_corr.detach().float().pow(2).mean().item())

        if c_sq > r_sq and r_sq > 0.0:
            scale = math.sqrt(r_sq / c_sq)
            v_corrected = self.v_mean + scale * delta_corr

        return v_corrected

# ═══════════════════════════════════════════════════════════════════════════════
# RF Trajectory Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_build_cache_from_sampler_sigmas(
    ref_clean:      torch.Tensor,
    sampler_sigmas: List[float],
    apply_model_fn: Callable,
    base_model_kwargs: Dict[str, Any],
    gamma:          float = 0.5,
    seed:           int   = 0,
    stats:          Optional[_RuntimeStats] = None,
    eps:            Optional[torch.Tensor] = None,
    rf_mode:        str   = 'rf_gamma',
    norm_strength:  float = 0.0,
    pmi_alpha:      float = 0.4,
    gamma_curve:     float = 0.0,
    preview_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]] = None,
) -> Tuple[Dict[float, torch.Tensor], torch.Tensor, List[float]]:
    norm_strength = _coerce_norm_strength(norm_strength)
    mode, gamma_curve = _normalize_rf_mode_and_gamma_curve(rf_mode, gamma_curve)
    parameterization = getattr(stats, 'parameterization', 'unknown') if stats else 'unknown'

    device = ref_clean.device
    dtype  = ref_clean.dtype

    if eps is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        eps = torch.randn(ref_clean.shape, device=device, dtype=dtype, generator=rng)

    sigmas: List[float] = [0.0]
    for s in sampler_sigmas:
        try:
            sf = max(0.0, min(1.0, float(s)))
        except Exception:
            continue
        if all(abs(sf - existing) > 1e-6 for existing in sigmas):
            sigmas.append(sf)
    sigmas = sorted(sigmas)

    z = ref_clean.clone()
    prev = 0.0
    cache: Dict[float, torch.Tensor] = {0.0: z.detach().clone()}
    model_ok = 0
    failures = 0
    vm_sum = 0.0
    vp_sum = 0.0
    next_step_velocity: Optional[torch.Tensor] = None

    pmi_alpha_eff = max(0.0, min(1.0, float(pmi_alpha)))
    use_pmi = pmi_alpha_eff > 0.0
    pmi_state = _PMIState()
    total_preview_steps = max(1, len(sigmas) - 1)
    previewed_steps: set = set()

    def _preview_once(step_index: int, raw_pred: Optional[torch.Tensor], x_current: Optional[torch.Tensor]) -> None:
        if preview_callback is None or raw_pred is None:
            return
        step_index = max(0, min(total_preview_steps - 1, int(step_index)))
        if step_index in previewed_steps:
            return
        previewed_steps.add(step_index)
        _rf_emit_preview(preview_callback, step_index, raw_pred, x_current, total_preview_steps)

    rf_total_steps = max(1, len(sigmas) - 1)
    rf_progress_start_time = time.time()

    for step_index in _rf_step_iterator(rf_total_steps):
        step_i = int(step_index) + 1
        s = sigmas[step_i]
        sigma_prev = float(prev)
        sigma_cur  = float(s)
        delta      = float(sigma_cur - sigma_prev)
        z_prev     = z.detach().clone()
        gamma_eff  = _rf_gamma_for_mode(mode, gamma, sigma_prev, sigma_cur, gamma_curve)
        vm_abs = 0.0
        vp_abs = 0.0
        extra  = ''

        def _call_model_as_velocity(z_in, sigma_val, label=''):
            nonlocal model_ok, failures, vm_sum
            t_tensor = torch.full((z_in.shape[0],), sigma_val, device=device, dtype=dtype)
            with torch.no_grad():
                try:
                    raw = apply_model_fn(z_in, t_tensor, **base_model_kwargs)
                    model_ok += 1
                    v = _velocity_from_pred(z_in, raw, sigma_val, parameterization)
                    vm_sum += float(v.abs().mean().item())
                    return v, True, raw
                except Exception as exc:
                    failures += 1
                    print(f'{_PREFIX}   [WARNING {mode}{label}] model failed at σ={sigma_val:.6f}: {exc}')
                    return torch.zeros_like(z_in), False, None

        def _apply_pmi_if_enabled(v: torch.Tensor) -> torch.Tensor:
            if not use_pmi:
                return v
            return pmi_state.update_and_correct(v, alpha=pmi_alpha_eff)

        if mode == 'linear':
            z = _rf_linear_target(ref_clean, eps, sigma_cur)
            extra = 'linear_target'
        elif mode == 'fireflow':
            if next_step_velocity is None:
                v_pred, ok, raw_preview = _call_model_as_velocity(z, sigma_prev, ' fresh')
                vm_abs = float(v_pred.abs().mean().item())
                pred_source = 'fresh'
            else:
                v_pred = next_step_velocity.to(device=device, dtype=dtype)
                vm_abs = float(v_pred.abs().mean().item())
                pred_source = 'reused_mid'

            z_mid      = z + 0.5 * delta * v_pred
            sigma_mid  = sigma_prev + 0.5 * delta
            v_mid, ok, raw_preview_mid = _call_model_as_velocity(z_mid, sigma_mid, ' mid')
            vm_abs_mid = float(v_mid.abs().mean().item())

            v_mid_total = _apply_pmi_if_enabled(v_mid)
            next_step_velocity = v_mid_total.detach().clone()
            z = z + delta * v_mid_total
            _preview_once(step_i - 1, raw_preview_mid, z)
            extra = f'FireFlow pred={pred_source} σ_mid={sigma_mid:.6f}'
        else:
            v_model, ok, raw_preview = _call_model_as_velocity(z, sigma_prev)
            vm_abs = float(v_model.abs().mean().item())

            denom   = max(1.0 - sigma_prev, 1e-7)
            v_prior = (eps - z) / denom
            vp_abs  = float(v_prior.abs().mean().item())
            vp_sum += vp_abs

            if mode == 'rf_gamma_rk2':
                v1    = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
                z_mid = z + 0.5 * delta * v1
                sigma_mid = sigma_prev + 0.5 * delta
                v_model_mid, ok_mid, raw_preview_mid = _call_model_as_velocity(z_mid, sigma_mid, ' mid')
                vm_abs_mid = float(v_model_mid.abs().mean().item())

                denom_mid = max(1.0 - sigma_mid, 1e-7)
                v_prior_mid = (eps - z_mid) / denom_mid
                vp_abs_mid  = float(v_prior_mid.abs().mean().item())
                vp_sum += vp_abs_mid

                v_total = gamma_eff * v_model_mid + (1.0 - gamma_eff) * v_prior_mid
                v_total = _apply_pmi_if_enabled(v_total)
                z = z + delta * v_total
                _preview_once(step_i - 1, raw_preview_mid, z)
                extra = f'mid |v_model_mid|={vm_abs_mid:.5f}'
            else:
                v_total = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
                v_total = _apply_pmi_if_enabled(v_total)
                z = z + delta * v_total
                _preview_once(step_i - 1, raw_preview, z)

        if norm_strength > 0.0:
            target = _rf_linear_target(ref_clean, eps, sigma_cur)
            z = _rf_match_mean_std(z, target, strength=norm_strength)
            extra = (extra + '  ' if extra else '') + f'norm={norm_strength:.2f}'

        prev  = sigma_cur
        cache[round(sigma_cur, 6)] = z.detach().clone()

        _rf_progress_snapshot(
            step_i,
            rf_total_steps,
            rf_progress_start_time,
            persistent=_coerce_bool(getattr(stats, 'rf_verbose', False)),
        )

    return cache, eps, sigmas

def _rf_increment_reference_one_step(
    z_prev:         torch.Tensor,
    sigma_prev:     float,
    sigma_cur:      float,
    apply_model_fn: Callable,
    base_model_kwargs: Dict[str, Any],
    gamma:          float = 0.5,
    seed:           int   = 0,
    stats:          Optional[_RuntimeStats] = None,
    eps:            Optional[torch.Tensor] = None,
    rf_mode:        str   = 'rf_gamma',
    gamma_curve: float = 0.0,
    norm_strength: float = 0.0,
    preview_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sigma_cur = max(0.0, min(1.0, float(sigma_cur)))
    delta_sigma = sigma_cur

    device = z_prev.device
    dtype  = z_prev.dtype
    parameterization = getattr(stats, 'parameterization', 'unknown') if stats else 'unknown'

    if eps is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        eps = torch.randn(z_prev.shape, device=device, dtype=dtype, generator=rng)

    t_tensor = torch.zeros((z_prev.shape[0],), device=device, dtype=dtype)
    raw = None
    with torch.no_grad():
        try:
            raw     = apply_model_fn(z_prev, t_tensor, **base_model_kwargs)
            v_model = _velocity_from_pred(z_prev, raw, 0.0, parameterization)
            model_ok = True
        except Exception as exc:
            print(f'{_PREFIX}   [WARNING direct RF] model call failed at σ=0.000000: {exc}')
            v_model = torch.zeros_like(z_prev)
            model_ok = False

    v_prior = eps - z_prev
    gamma_eff = _rf_gamma_for_mode(rf_mode, gamma, sigma_prev, sigma_cur, gamma_curve)
    v_total = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
    z_cur   = z_prev + delta_sigma * v_total
    norm_strength = max(0.0, min(1.0, float(norm_strength)))
    if norm_strength > 0.0:
        target = _rf_linear_target(z_prev, eps, sigma_cur)
        z_cur = _rf_match_mean_std(z_cur, target, strength=norm_strength)

    _rf_emit_preview(preview_callback, 0, raw, z_cur, 1)
    return z_cur, eps

def _normalize_sigma_float(value: Any) -> Optional[float]:
    try:
        if torch.is_tensor(value):
            viewer = float(value.detach().float().mean().item())
        else:
            viewer = float(value)
        if not math.isfinite(viewer):
            return None
        if 0.0 <= viewer <= 1.0:
            return max(0.0, min(1.0, viewer))
        if 1.0 < viewer <= 1000.0:
            return max(0.0, min(1.0, viewer / 1000.0))
    except Exception:
        return None
    return None

def _coerce_sigma_sequence(value: Any) -> Optional[List[float]]:
    try:
        if value is None:
            return None
        if torch.is_tensor(value):
            flat = value.detach().float().flatten().tolist()
        elif isinstance(value, (list, tuple)):
            flat = []
            for item in value:
                if torch.is_tensor(item):
                    flat.extend(item.detach().float().flatten().tolist())
                elif isinstance(item, (int, float)):
                    flat.append(float(item))
                else:
                    return None
        else:
            return None
        out: List[float] = []
        for item in flat:
            s = _normalize_sigma_float(item)
            if s is not None:
                out.append(s)
        if len(out) < 2:
            return None
        dedup: List[float] = []
        for s in out:
            if not dedup or abs(dedup[-1] - s) > 1e-6:
                dedup.append(s)
        return dedup if len(dedup) >= 2 else None
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# Comfy Model Extraction Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_get_diffusion_model(model_patcher: Any) -> Any:
    if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
        return model_patcher.model.diffusion_model
    if hasattr(model_patcher, 'diffusion_model'):
        return model_patcher.diffusion_model
    raise RuntimeError('Could not find standard diffusion model.')

def _repeat_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if comfy is not None and hasattr(comfy.utils, 'repeat_to_batch_size'):
        return comfy.utils.repeat_to_batch_size(x, batch)
    reps = math.ceil(batch / x.shape[0])
    return x.repeat((reps,) + (1,) * (x.ndim - 1))[:batch]

def _clone_model_options(options: Dict[str, Any]) -> Dict[str, Any]:
    out = options.copy()
    out['transformer_options'] = options.get('transformer_options', {}).copy()
    return out

def _clone_conditioning_for_rf(c: Dict[str, Any]) -> Dict[str, Any]:
    out = c.copy()
    to  = out.get('transformer_options', {})
    if isinstance(to, dict):
        to = to.copy()
        to.pop('untwisting_rope_zimage', None)
        out['transformer_options'] = to
    else:
        out['transformer_options'] = {}
    return out

def _slice_conditioning_batch(obj: Any, start: int, end: int) -> Any:
    if torch.is_tensor(obj):
        try:
            if obj.ndim > 0 and int(obj.shape[0]) >= end:
                return obj[start:end]
        except Exception:
            pass
        return obj
    if isinstance(obj, dict):
        return {
            k: (v if k == 'transformer_options' else _slice_conditioning_batch(v, start, end))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_slice_conditioning_batch(v, start, end) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_slice_conditioning_batch(v, start, end) for v in obj)
    return obj

def _build_rf_conditioning_kwargs(
    c: Dict[str, Any],
    ref_conditioning: Any,
    target_b: int,
) -> Tuple[Dict[str, Any], str]:
    try:
        if ref_conditioning is None:
            ref_conditioning = c
        merged, _forced = _merge_reference_conditioning_into_c(c, ref_conditioning, target_b)
        ref_only = _slice_conditioning_batch(merged, target_b, target_b * 2)
        return _clone_conditioning_for_rf(ref_only), 'reference'
    except Exception as exc:
        print(f'{_PREFIX}   ⚠ RF conditioning fallback: {exc}')
    return _clone_conditioning_for_rf(c), 'target-fallback'

def _sigma_from_timestep(timestep: torch.Tensor) -> float:
    try:
        val = float(timestep.detach().float().mean().item())
        if math.isfinite(val):
            if 0.0 <= val <= 1.0:
                return max(0.0, min(1.0, val))
            if 1.0 < val <= 1000.0:
                return max(0.0, min(1.0, val / 1000.0))
    except Exception:
        pass
    return 1.0

def _sigma_to_progress(timestep: torch.Tensor) -> float:
    return max(0.0, min(1.0, 1.0 - _sigma_from_timestep(timestep)))

# ═══════════════════════════════════════════════════════════════════════════════
# Reference Conditioning Integration
# ═══════════════════════════════════════════════════════════════════════════════

_TEXT_CONDITIONING_KEYS = {'c_crossattn', 'crossattn', 'context', 'cap_feats', 'cond'}
_POOLED_CONDITIONING_KEYS = {'pooled_output', 'clip_pooled', 'pooled', 'y', 'vector'}
_MASK_CONDITIONING_KEYS = {'attention_mask', 'crossattn_mask', 'c_crossattn_mask'}
_NUM_TOKEN_KEYS = {'num_tokens', 'tokens_num', 'n_tokens'}

def _first_tensor_in_conditioning_entry(entry: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if torch.is_tensor(entry):
        return entry, meta
    if isinstance(entry, dict):
        meta.update(entry)
        for key in ('c_crossattn', 'crossattn', 'conditioning', 'cond', 'context', 'cap_feats'):
            value = entry.get(key)
            if torch.is_tensor(value):
                return value, meta
        for value in entry.values():
            if torch.is_tensor(value) and value.ndim >= 2:
                return value, meta
        return None, meta
    return None, meta

def _extract_reference_conditioning(ref_conditioning: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if ref_conditioning is None:
        return None, {}
    if torch.is_tensor(ref_conditioning) or isinstance(ref_conditioning, dict):
        return _first_tensor_in_conditioning_entry(ref_conditioning)
    if isinstance(ref_conditioning, (list, tuple)):
        merged_meta: Dict[str, Any] = {}
        for entry in ref_conditioning:
            cond, meta = _first_tensor_in_conditioning_entry(entry)
            if meta:
                merged_meta.update(meta)
            if cond is not None:
                return cond, merged_meta
    return None, {}

def _merge_reference_conditioning_into_c(c, ref_conditioning, target_b):
    if ref_conditioning is None or ref_conditioning is c:
        out: Dict[str, Any] = {}
        for key, value in c.items():
            if key == 'transformer_options':
                out[key] = value
                continue
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == target_b:
                out[key] = torch.cat([value, value], dim=0)
            else:
                out[key] = value
                
        for key in ('c_crossattn', 'crossattn', 'context', 'cap_feats', 'cond'):
            val = c.get(key)
            if torch.is_tensor(val):
                forced_cap_mask = torch.ones((target_b * 2, val.shape[1]), device=val.device, dtype=torch.bool)
                return out, forced_cap_mask
        forced_cap_mask = torch.ones((target_b * 2, 512), device=torch.device("cpu"), dtype=torch.bool)
        return out, forced_cap_mask

    ref_cond, ref_meta = _extract_reference_conditioning(ref_conditioning)
    if ref_cond is None:
        return _merge_reference_conditioning_into_c(c, None, target_b)

    target_text = None
    for key in ('c_crossattn', 'crossattn', 'context', 'cap_feats', 'cond'):
        value = c.get(key)
        if torch.is_tensor(value) and value.ndim >= 3 and int(value.shape[0]) == target_b:
            target_text = value
            break

    if target_text is None:
        return _merge_reference_conditioning_into_c(c, None, target_b)

    if ref_cond.ndim == target_text.ndim - 1:
        ref_cond = ref_cond.unsqueeze(0)

    out: Dict[str, Any] = {}
    for key, value in c.items():
        if key == 'transformer_options':
            out[key] = value
            continue
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == target_b:
            if key in _TEXT_CONDITIONING_KEYS:
                ref_part = _repeat_to_batch(ref_cond, target_b)
                if ref_part.shape[1] != value.shape[1]:
                    pad_len = max(ref_part.shape[1], value.shape[1])
                    if value.shape[1] < pad_len:
                        value = torch.nn.functional.pad(value, (0, 0, 0, pad_len - value.shape[1]))
                    if ref_part.shape[1] < pad_len:
                        ref_part = torch.nn.functional.pad(ref_part, (0, 0, 0, pad_len - ref_part.shape[1]))
                out[key] = torch.cat([value, ref_part], dim=0)
            elif key in _POOLED_CONDITIONING_KEYS:
                ref_val = ref_meta.get(key, value)
                if torch.is_tensor(ref_val):
                    ref_part = _repeat_to_batch(ref_val, target_b).to(device=value.device, dtype=value.dtype)
                    out[key] = torch.cat([value, ref_part], dim=0)
                else:
                    out[key] = torch.cat([value, value], dim=0)
            else:
                out[key] = torch.cat([value, value], dim=0)
        else:
            out[key] = value

    forced_cap_mask = torch.ones((target_b * 2, target_text.shape[1]), device=target_text.device, dtype=torch.bool)
    return out, forced_cap_mask

# ═══════════════════════════════════════════════════════════════════════════════
# Position frequency scaling logic
# ═══════════════════════════════════════════════════════════════════════════════

def _build_frequency_scale_vector(
    head_dim, axes_dims, high_scale, low_scale, beta, device, dtype
):
    if not axes_dims or sum(int(x) for x in axes_dims) != head_dim:
        axes_dims = [head_dim]
    axes_dims = [int(x) for x in axes_dims]
    is_3axis  = len(axes_dims) == 3
    pieces: List[torch.Tensor] = []
    for axis_idx, axis_dim in enumerate(axes_dims):
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue
        if is_3axis and axis_idx == 0:
            pair_scales = torch.full(
                (n_pairs,), float(low_scale), device=device, dtype=torch.float32
            )
        else:
            d_tilde = (
                torch.zeros(1, device=device, dtype=torch.float32)
                if n_pairs == 1
                else torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
            )
            pair_scales = high_scale + (low_scale - high_scale) * d_tilde.pow(float(beta))
        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))
    out = torch.cat(pieces, dim=0)
    if out.numel() >= head_dim:
        return out[:head_dim]
    return torch.nn.functional.pad(out, (0, head_dim - out.numel()), value=1.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Nodes Setup
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_new_debug_store() -> Dict[str, Any]:
    debug_store: Dict[str, Any] = _RF_LAST_DEBUG_STORE
    debug_store.clear()
    debug_store.update({
        'cache': {},
        'xhat_cache': {},
        'pred_cache': {},
        'xhat_plus_cache': {},
        'sampler_sigmas': None,
        'built_sigmas': None,
        'run_count': 0,
        'persistent_cache_key': None,
        'persistent_cache_hit': False,
        'parameterization': 'unknown',
    })
    return debug_store

def _rf_make_preview_callback(model_for_preview: Any, total_steps: int) -> Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]]:
    total_steps = max(1, int(total_steps))
    if latent_preview is not None:
        try:
            return latent_preview.prepare_callback(model_for_preview, total_steps)
        except Exception as exc:
            print(f'{_PREFIX} ⚠ RF preview callback disabled: {exc}')
    try:
        pbar = comfy.utils.ProgressBar(total_steps)
        def _progress_only(step: int, x0: torch.Tensor, x: torch.Tensor, steps: int) -> None:
            pbar.update_absolute(step + 1, steps)
        return _progress_only
    except Exception as exc:
        print(f'{_PREFIX} ⚠ RF progress callback disabled: {exc}')
        return None

def _rf_emit_preview(
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]],
    step: int,
    raw_pred: Optional[torch.Tensor],
    x_current: Optional[torch.Tensor],
    total_steps: int,
) -> None:
    if callback is None or not torch.is_tensor(raw_pred):
        return
    try:
        preview_latent = raw_pred[:1].detach()
        current = x_current[:1].detach() if torch.is_tensor(x_current) else preview_latent
        callback(int(step), preview_latent, current, int(total_steps))
    except Exception as exc:
        print(f'{_PREFIX} ⚠ RF preview frame failed at step {int(step) + 1}: {exc}')

def _rf_latent_get_config(rf_inversion: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[torch.Tensor], Optional[Any], str]:
    if not isinstance(rf_inversion, dict):
        return False, {}, {}, None, None, 'not-connected'
    cfg = rf_inversion.get('untwist_rf_config', None)
    state = rf_inversion.get('untwist_rf_state', None)
    ref_clean = rf_inversion.get('untwist_ref_clean', None)
    ref_conditioning = rf_inversion.get('untwist_ref_conditioning', None)
    if not isinstance(cfg, dict):
        return False, {}, {}, None, None, 'missing-config'
    if not isinstance(state, dict):
        state = {}
        rf_inversion['untwist_rf_state'] = state
    if not torch.is_tensor(ref_clean):
        return False, cfg, state, None, ref_conditioning, 'missing-ref-clean'
    return True, cfg, state, ref_clean, ref_conditioning, 'RFInversionK9b LATENT'

class RFInversionK9b:
    CATEGORY = 'model_patches/Untwisting RoPE-Klein9b'
    RETURN_TYPES = ('MODEL', 'LATENT')
    RETURN_NAMES = ('model', 'rf_inversion')
    FUNCTION = 'build'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'reference_latent': ('LATENT',),
                'rf_mode': (['linear', 'rf_gamma', 'rf_gamma_rk2', 'fireflow'], {
                    'default': 'rf_gamma',
                }),
                'gamma': ('FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'gamma_curve': ('FLOAT', {'default': 2.0, 'min': 0.0, 'max': 8.0, 'step': 0.05}),
                'norm_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
                'pmi_alpha': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
                'verbose': ('BOOLEAN', {'default': False}),
            },
            'optional': {
                'ref_conditioning': ('CONDITIONING',),
            },
        }

    def build(
        self,
        model,
        reference_latent,
        rf_mode='fireflow',
        gamma=0.3,
        gamma_curve=0.0,
        norm_strength=0.0,
        pmi_alpha=0.4,
        verbose=False,
        ref_conditioning=None,
    ):
        rf_mode, gamma_curve = _normalize_rf_mode_and_gamma_curve(rf_mode, gamma_curve)
        norm_strength = _coerce_norm_strength(norm_strength)
        verbose_flag = _coerce_bool(verbose)

        if not isinstance(reference_latent, dict) or 'samples' not in reference_latent:
            raise RuntimeError("reference_latent must contain a 'samples' latent tensor.")

        ref_clean = reference_latent['samples'].detach().clone()
        ref_clean = model.model.process_latent_in(ref_clean)

        detected_param = 'unknown'
        try:
            m_type = str(getattr(model.model, 'model_type', ''))
            if 'FLOW' in m_type:
                detected_param = 'flow_velocity'
            elif 'V_PREDICTION' in m_type:
                detected_param = 'velocity'
            elif 'X0' in m_type:
                detected_param = 'x0'
        except Exception:
            pass

        cfg: Dict[str, Any] = {
            'rf_mode': str(rf_mode),
            'gamma': float(gamma),
            'gamma_curve': float(gamma_curve),
            'norm_strength': float(norm_strength),
            'pmi_alpha': float(pmi_alpha),
            'seed': 42,
            'verbose': verbose_flag,
        }
        state: Dict[str, Any] = {
            'cache': {0.0: ref_clean.detach().to(device='cpu').clone()},
            'eps': None,
            'prev_z': None,
            'prev_sigma': None,
            'run_count': 0,
            'sampler_sigmas': None,
            'schedule_built': False,
            'schedule_sorted': None,
            'persistent_cache_key': None,
            'persistent_cache_hit': False,
            'preview_callback': None,
        }
        debug_store = _rf_new_debug_store()
        debug_store['cache'] = state['cache']
        debug_store['parameterization'] = detected_param

        rf_latent: Dict[str, Any] = dict(reference_latent)
        rf_latent['samples'] = reference_latent['samples']
        rf_latent['untwist_rf_config'] = cfg
        rf_latent['untwist_rf_state'] = state
        rf_latent['untwist_rf_cache'] = state['cache']
        rf_latent['untwist_rf_sigmas'] = None
        rf_latent['untwist_rf_mode'] = str(rf_mode)
        rf_latent['untwist_rf_seed'] = 42
        rf_latent['untwist_rf_parameterization'] = detected_param
        rf_latent['untwist_ref_clean'] = ref_clean.detach().to(device='cpu').clone()
        rf_latent['untwist_ref_conditioning'] = ref_conditioning

        model_clone = model.clone()
        setattr(model_clone, '_untwisting_rope_rf_debug', debug_store)
        setattr(model_clone, '_untwisting_rope_rf_state', state)
        setattr(model_clone, '_untwisting_rope_rf_config', cfg)

        def sampler_sample_wrapper(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            found = _coerce_sigma_sequence(sigmas)
            if found is not None:
                state['sampler_sigmas'] = found
                state['schedule_built'] = False
                state['schedule_sorted'] = None
                state['persistent_cache_key'] = None
                state['persistent_cache_hit'] = False
                state['cache'] = {0.0: ref_clean.detach().to(device='cpu').clone()}
                state['eps'] = None
                state['run_count'] = int(state.get('run_count', 0)) + 1
                try:
                    state['preview_callback'] = _rf_make_preview_callback(model_clone, max(1, len(found) - 1))
                except Exception:
                    state['preview_callback'] = None

                rf_latent['untwist_rf_cache'] = state['cache']
                rf_latent['untwist_rf_sigmas'] = list(found)
                rf_latent['untwist_rf_state'] = state

                debug_store['cache'] = state['cache']
                debug_store['sampler_sigmas'] = list(found)
                debug_store['built_sigmas'] = None
                debug_store['run_count'] = int(state['run_count'])
                debug_store['persistent_cache_key'] = None
                debug_store['persistent_cache_hit'] = False
                debug_store['parameterization'] = rf_latent.get('untwist_rf_parameterization', 'unknown')
            return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

        model_clone.model_options = _clone_model_options(model_clone.model_options)
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            sampler_sample_wrapper,
            model_clone.model_options,
            is_model_options=True,
        )

        return (model_clone, rf_latent)

class UntwistingRoPEK9b:
    CATEGORY = 'model_patches/Untwisting RoPE'
    RETURN_TYPES = ('MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'patch'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'beta': ('FLOAT', {'default': 50.0, 'min': 0.01, 'max': 100.0, 'step': 0.01}),
                'high_scale_start': ('FLOAT', {'default': 1.0, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'high_scale_end': ('FLOAT', {'default': 0.00, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'low_scale_start': ('FLOAT', {'default': 1.0, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'low_scale_end': ('FLOAT', {'default': 8.0, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'start_block': ('INT', {'default': 1, 'min': 0, 'max': 999, 'step': 1}),
                'end_block': ('INT', {'default': 999, 'min': 0, 'max': 999, 'step': 1}),
                'adain_strength': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'verbose': ('BOOLEAN', {'default': False}),
            },
            'optional': {
                'rf_inversion': ('LATENT',),
            },
        }

    def patch(
        self,
        model,
        beta: float,
        high_scale_start: float,
        high_scale_end: float,
        low_scale_start: float,
        low_scale_end: float,
        start_block: int,
        end_block: int,
        adain_strength: float,
        verbose: bool = False,
        rf_inversion: Optional[Dict[str, Any]] = None,
    ):
        rf_active, rf_cfg, rf_state, ref_clean_cpu, ref_conditioning, rf_source = _rf_latent_get_config(rf_inversion)
        node_verbose = _coerce_bool(verbose)
        rf_verbose = _coerce_bool(rf_cfg.get('verbose', False))
        stats = _RuntimeStats(verbose=node_verbose, rf_verbose=rf_verbose)
        debug_store = _rf_new_debug_store()

        rf_mode = str(rf_cfg.get('rf_mode', 'fireflow'))
        gamma = float(rf_cfg.get('gamma', 0.3))
        gamma_curve = float(rf_cfg.get('gamma_curve', 0.0))
        norm_strength = float(rf_cfg.get('norm_strength', 0.0))
        pmi_alpha = float(rf_cfg.get('pmi_alpha', 0.4))
        seed = int(rf_cfg.get('seed', 42))

        if rf_active:
            stats.rf_sigma_cache = rf_state.get('cache', {}) if isinstance(rf_state.get('cache', {}), dict) else {}
            stats.rf_schedule_built = bool(rf_state.get('schedule_built', False))
            stats.parameterization = str(rf_inversion.get('untwist_rf_parameterization', 'unknown')) if isinstance(rf_inversion, dict) else 'unknown'
            debug_store['cache'] = stats.rf_sigma_cache
            debug_store['sampler_sigmas'] = list(rf_state.get('sampler_sigmas') or []) if isinstance(rf_state, dict) else []
            debug_store['built_sigmas'] = list(rf_state.get('schedule_sorted') or []) if isinstance(rf_state, dict) else []
            debug_store['parameterization'] = stats.parameterization

        model_clone = model.clone()
        setattr(model_clone, '_untwisting_rope_rf_debug', debug_store)
        if rf_active:
            setattr(model_clone, '_untwisting_rope_rf_state', rf_state)
            setattr(model_clone, '_untwisting_rope_rf_config', rf_cfg)

        old_wrapper = model_clone.model_options.get('model_function_wrapper', None)

        def model_function_wrapper(apply_model: Callable, args: Dict[str, Any]) -> torch.Tensor:
            stats.wrapper_calls += 1
            call_n = stats.wrapper_calls

            input_x = args['input']
            timestep = args['timestep']
            c = args['c'].copy()
            cond_or_uncond = args.get('cond_or_uncond', None)
            to = c.get('transformer_options', {}).copy()

            sigma = _sigma_from_timestep(timestep)
            progress = _sigma_to_progress(timestep)
            target_b = int(input_x.shape[0])

            axes_dims = [16, 56, 56]
            try:
                dm = _safe_get_diffusion_model(model_clone)
                if hasattr(dm, "pe_embedder") and hasattr(dm.pe_embedder, "axes_dim"):
                    axes_dims = list(dm.pe_embedder.axes_dim)
            except Exception:
                pass

            cfg: Dict[str, Any] = {
                'enabled': True,
                'beta': float(beta),
                'high_scale_start': float(high_scale_start),
                'high_scale_end': float(high_scale_end),
                'low_scale_start': float(low_scale_start),
                'low_scale_end': float(low_scale_end),
                'start_block': int(start_block),
                'end_block': int(end_block),
                'apply_adain': True,
                'adain_strength': float(adain_strength),
                'cross_batch_target_batch': target_b if rf_active else 0,
                'progress': progress,
                'axes_dims': axes_dims,
            }
            to['untwisting_rope_zimage'] = cfg

            input_for_model = input_x
            timestep_for_model = timestep
            ref_noisy = None
            sigma_key = round(float(sigma), 6)
            rf_cache_hit = False
            rf_cond_mode = 'not-connected'
            ref_mode = 'target-only'

            if rf_active and torch.is_tensor(ref_clean_cpu):
                try:
                    ref_clean = ref_clean_cpu.to(device=input_x.device, dtype=input_x.dtype)
                    ref = _repeat_to_batch(ref_clean, target_b)

                    if not rf_state.get('schedule_built', False) and rf_state.get('sampler_sigmas', None) is not None:
                        rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, ref_conditioning, target_b)
                        rf_ref_clean = _repeat_to_batch(ref_clean, target_b)
                        sampler_sigmas = list(rf_state['sampler_sigmas'])
                        cache_key = _make_rf_persistent_key(
                            ref_clean=ref_clean_cpu.detach().to(device='cpu'),
                            ref_conditioning=ref_conditioning,
                            sampler_sigmas=sampler_sigmas,
                            target_b=target_b,
                            rf_mode=rf_mode,
                            gamma=gamma,
                            gamma_curve=gamma_curve,
                            norm_strength=norm_strength,
                            cond_mode=rf_cond_mode,
                            pmi_alpha=pmi_alpha,
                        )
                        cached_entry = _RF_PERSISTENT_TRAJECTORY_CACHE.get(cache_key)
                        if cached_entry is not None:
                            built_cache = _cache_to_device(cached_entry['cache'], input_x.device, input_x.dtype)
                            eps = cached_entry['eps'].to(device=input_x.device, dtype=input_x.dtype)
                            sorted_sigmas = list(cached_entry['built_sigmas'])
                            ref_mode = 'RF persistent cache'
                            rf_state['persistent_cache_hit'] = True
                        else:
                            preview_callback = rf_state.get('preview_callback', None)
                            if preview_callback is None:
                                preview_callback = _rf_make_preview_callback(model_clone, max(1, len(sampler_sigmas) - 1))
                                rf_state['preview_callback'] = preview_callback
                            built_cache, eps, sorted_sigmas = _rf_build_cache_from_sampler_sigmas(
                                ref_clean=rf_ref_clean,
                                sampler_sigmas=sampler_sigmas,
                                apply_model_fn=apply_model,
                                base_model_kwargs=rf_kwargs,
                                gamma=gamma,
                                seed=seed,
                                stats=stats,
                                eps=rf_state['eps'].to(device=input_x.device, dtype=input_x.dtype)
                                    if torch.is_tensor(rf_state.get('eps', None)) else None,
                                rf_mode=rf_mode,
                                gamma_curve=gamma_curve,
                                norm_strength=norm_strength,
                                pmi_alpha=pmi_alpha,
                                preview_callback=preview_callback,
                            )
                            _put_persistent_rf_cache(cache_key, {
                                'cache': _cache_to_cpu(built_cache),
                                'eps': eps.detach().to(device='cpu').clone(),
                                'built_sigmas': list(sorted_sigmas),
                            })
                            ref_mode = 'RF build successful'
                            rf_state['persistent_cache_hit'] = False

                        rf_state['cache'] = built_cache
                        rf_state['eps'] = eps.detach().clone()
                        rf_state['schedule_sorted'] = sorted_sigmas
                        rf_state['schedule_built'] = True
                        rf_state['persistent_cache_key'] = cache_key
                        stats.rf_sigma_cache = rf_state['cache']
                        stats.rf_eps = rf_state['eps']
                        stats.rf_schedule_built = True
                        stats.rf_step_count = max(0, len(sorted_sigmas) - 1)

                        if isinstance(rf_inversion, dict):
                            rf_inversion['untwist_rf_cache'] = _cache_to_cpu(built_cache)
                            rf_inversion['untwist_rf_eps'] = eps.detach().to(device='cpu').clone()
                            rf_inversion['untwist_rf_sigmas'] = list(sorted_sigmas)
                            rf_inversion['untwist_rf_state'] = rf_state

                        debug_store['cache'] = rf_state['cache']
                        debug_store['sampler_sigmas'] = list(rf_state.get('sampler_sigmas') or [])
                        debug_store['built_sigmas'] = list(sorted_sigmas)
                        debug_store['run_count'] = int(rf_state.get('run_count', 0))
                        debug_store['persistent_cache_key'] = cache_key
                        debug_store['persistent_cache_hit'] = bool(rf_state.get('persistent_cache_hit', False))
                        debug_store['parameterization'] = stats.parameterization
                    elif rf_state.get('schedule_built', False):
                        ref_mode = 'RF cache reference'
                    else:
                        rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, ref_conditioning, target_b)
                        rf_ref_clean = _repeat_to_batch(ref_clean, target_b)
                        preview_callback = rf_state.get('preview_callback', None)
                        if preview_callback is None:
                            preview_callback = _rf_make_preview_callback(model_clone, 1)
                            rf_state['preview_callback'] = preview_callback
                        z_sigma, eps = _rf_increment_reference_one_step(
                            z_prev=rf_ref_clean,
                            sigma_prev=0.0,
                            sigma_cur=sigma,
                            apply_model_fn=apply_model,
                            base_model_kwargs=rf_kwargs,
                            gamma=gamma,
                            seed=seed,
                            stats=stats,
                            eps=rf_state['eps'].to(device=input_x.device, dtype=input_x.dtype)
                                if torch.is_tensor(rf_state.get('eps', None)) else None,
                            rf_mode=rf_mode,
                            gamma_curve=gamma_curve,
                            norm_strength=norm_strength,
                            preview_callback=preview_callback,
                        )
                        rf_state['eps'] = eps.detach().clone()
                        cache = rf_state.get('cache') if isinstance(rf_state.get('cache'), dict) else {}
                        cache[sigma_key] = z_sigma.detach().clone()
                        rf_state['cache'] = cache
                        stats.rf_eps = rf_state['eps']
                        stats.rf_sigma_cache = rf_state['cache']
                        ref_mode = 'RF step direct'

                    cache = rf_state.get('cache') if isinstance(rf_state.get('cache'), dict) else {}
                    cached = cache.get(sigma_key, None)
                    if cached is None:
                        keys = [k for k in cache.keys() if isinstance(k, float)]
                        if keys:
                            nearest = min(keys, key=lambda k: abs(k - sigma_key))
                            cached = cache[nearest]
                            ref_mode += f' nearest({nearest:.6f})'
                        else:
                            cached = ref
                    else:
                        rf_cache_hit = True

                    ref_noisy = _repeat_to_batch(cached.to(device=input_x.device, dtype=input_x.dtype), target_b)

                    if ref_noisy.shape[-2:] == input_x.shape[-2:]:
                        input_for_model = torch.cat([input_x, ref_noisy], dim=0)
                        try:
                            if torch.is_tensor(timestep) and timestep.ndim > 0 and int(timestep.shape[0]) == target_b:
                                timestep_for_model = torch.cat([timestep, timestep], dim=0)
                            else:
                                timestep_for_model = _repeat_to_batch(timestep, target_b * 2)
                        except Exception:
                            timestep_for_model = timestep

                        c, forced_cap_mask = _merge_reference_conditioning_into_c(c, ref_conditioning, target_b)
                        cfg['forced_cap_mask'] = forced_cap_mask.to(device=input_x.device)
                        cfg['cross_batch_target_batch'] = target_b

                        try:
                            if isinstance(cond_or_uncond, list):
                                cond_or_uncond = cond_or_uncond + cond_or_uncond
                        except Exception:
                            pass
                    else:
                        cfg['enabled'] = False
                        cfg['cross_batch_target_batch'] = 0
                        ref_noisy = None
                except Exception as exc:
                    print(f'{_PREFIX} ⚠ RF guidance pipeline fallback: {exc}')
                    cfg['enabled'] = False
                    cfg['cross_batch_target_batch'] = 0
                    ref_noisy = None

            c['transformer_options'] = to

            if old_wrapper is not None:
                raw_result = old_wrapper(apply_model, {
                    'input': input_for_model,
                    'timestep': timestep_for_model,
                    'c': c,
                    'cond_or_uncond': cond_or_uncond,
                })
            else:
                raw_result = apply_model(input_for_model, timestep_for_model, **c)

            if rf_active and ref_noisy is not None and torch.is_tensor(raw_result) and raw_result.shape[0] >= target_b * 2:
                target_pred = raw_result[:target_b]
                ref_pred = raw_result[target_b:target_b * 2]

                try:
                    ref_xsigma = ref_noisy[:target_b]
                    debug_store['pred_cache'][sigma_key] = ref_pred[:1].detach().clone()
                    debug_store['xhat_cache'][sigma_key] = (ref_xsigma - float(sigma) * ref_pred)[:1].detach().clone()
                    debug_store['xhat_plus_cache'][sigma_key] = (ref_xsigma + float(sigma) * ref_pred)[:1].detach().clone()
                except Exception as exc:
                    print(f'{_PREFIX} ⚠ Failed to compute debug caches: {exc}')

                return target_pred

            return raw_result

        model_clone.model_options = _clone_model_options(model_clone.model_options)
        model_clone.set_model_unet_function_wrapper(model_function_wrapper)

        return (model_clone,)

NODE_CLASS_MAPPINGS = {
    'RFInversionK9b': RFInversionK9b,
    'UntwistingRoPEK9b': UntwistingRoPEK9b,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'RFInversionK9b': 'RF Inversion (Klein 9b)',
    'UntwistingRoPEK9b': 'Untwisting RoPE (Klein 9b)',
}
#!/usr/bin/env python3
"""Shared setup for the LLM-side VCPO experiments.

Everything that touches disk is pinned to the data volume. The system volume on
this host is 30 GB and the Hugging Face cache defaults to ``~/.cache``, which
lives there, so a single model download is enough to fill it. Import this
module before ``transformers`` in every entry point.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Disk pinning. Must happen before transformers is imported anywhere. -----
DATA_ROOT = Path(os.environ.get("EXPERIMENT_ROOT", "."))
MODELS_DIR = DATA_ROOT / "models"
RESULTS_DIR = DATA_ROOT / "results"
LOGS_DIR = DATA_ROOT / "logs"
# Model cache may live on a different volume than the results dir (e.g. models
# on the local volume, results on the shared volume). Respect an explicit
# HF_HOME when the caller sets it; otherwise pin it under EXPERIMENT_ROOT as before.
HF_CACHE = Path(os.environ.get("HF_HOME", str(DATA_ROOT / "hf_cache")))

for _path in (MODELS_DIR, RESULTS_DIR, LOGS_DIR, HF_CACHE):
    _path.mkdir(parents=True, exist_ok=True)

# huggingface.co is unreachable from this host; route everything through the mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Newer huggingface_hub defaults to Xet transfers, whose CAS server 401s behind
# the mirror; force plain HTTP downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))
# Pin temp files to the data volume: model loading (accelerate/device_map) and
# tokenizer merges write temp files that otherwise fill the 30G system volume.
_TMP = DATA_ROOT / "tmp"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_TMP))
os.environ.setdefault("TEMP", str(_TMP))
os.environ.setdefault("TMP", str(_TMP))
# Tokenizer parallelism warns loudly once we fork for the tabular sweep.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def disk_guard(minimum_free_gb: float = 5.0) -> None:
    """Abort early if the system volume is close to full.

    Filling the system volume wedges the container, so this is checked before
    any download rather than discovered halfway through one.
    """

    usage = os.statvfs("/")
    free_gb = usage.f_bavail * usage.f_frsize / 1e9
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"system volume has only {free_gb:.1f} GB free; "
            f"check that HF_HOME points at {HF_CACHE}"
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunMeta:
    """Metadata recorded with every measurement.

    The paper reports slopes across models, datasets and length units, and a
    number without these fields cannot be placed in any of those tables. They
    are written next to the numbers rather than reconstructed later.
    """

    experiment: str
    model: str
    dataset: str
    seed: int
    device: str = ""
    dtype: str = ""
    torch_version: str = ""
    transformers_version: str = ""
    extra: dict = field(default_factory=dict)

    def fill_environment(self) -> "RunMeta":
        import transformers

        self.device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.torch_version = torch.__version__
        self.transformers_version = transformers.__version__
        return self


def write_result(name: str, meta: RunMeta, payload: dict) -> Path:
    """Write one result file, metadata first."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with path.open("w") as handle:
        json.dump({"meta": asdict(meta), **payload}, handle, indent=2, sort_keys=True)
    return path


def load_model(model_id: str, dtype: torch.dtype = torch.bfloat16, train: bool = False):
    """Load a causal LM and its tokenizer onto the data volume's cache.

    ``train=False`` puts the model in eval mode with gradients disabled, which
    is what the log-probability measurements need; ``train=True`` leaves
    gradients on for the gradient-alignment experiments.
    """

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

    # transformers>=5 requires all_tied_weights_keys on models; some remote-code
    # models (e.g. GLM-4-9b) only expose the private _tied_weights_keys, so
    # provide a fallback on the base class (built-ins with their own property
    # still shadow it via the MRO).
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

        def _get_all_tied(self):
            return getattr(self, "_all_tied_weights_keys", None) or {}

        def _set_all_tied(self, value):
            self._all_tied_weights_keys = value

        PreTrainedModel.all_tied_weights_keys = property(_get_all_tied, _set_all_tied)

    disk_guard()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(HF_CACHE / "hub"), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(
        model_id, cache_dir=str(HF_CACHE / "hub"), trust_remote_code=True
    )
    # GLM-4-9b's remote config uses ChatGLM-era attribute names while
    # transformers>=5 (generation/cache code) reads the standard names. Alias the
    # standard ones onto the config. These are architecture params, not
    # generation controls, so they can stay.
    for _dst, _src in [
        ("num_hidden_layers", "num_layers"),
        ("num_key_value_heads", "multi_query_group_num"),
        ("head_dim", "kv_channels"),
        ("max_position_embeddings", "seq_length"),
        ("rms_norm_eps", "layernorm_epsilon"),
    ]:
        if not hasattr(config, _dst) and hasattr(config, _src):
            setattr(config, _dst, getattr(config, _src))
    # max_length is special: GLM __init__ reads it, but transformers>=5 treats a
    # lingering config.max_length as a generation-control override and raises in
    # generate(). Alias it only for construction, then drop it.
    alias_max_length = (not hasattr(config, "max_length")) and hasattr(config, "seq_length")
    if alias_max_length:
        config.max_length = config.seq_length
    if not hasattr(config, "use_cache"):
        config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=str(HF_CACHE / "hub"),
        trust_remote_code=True,
    )
    if alias_max_length:
        for cfg in (config, getattr(model, "config", None)):
            if cfg is not None and hasattr(cfg, "max_length"):
                try:
                    del cfg.max_length
                except AttributeError:
                    pass
    if train:
        model.train()
        # Gradient checkpointing trades compute for activation memory, which is
        # what lets a 14B model do a backward pass inside an 80 GB GPU.
        model.gradient_checkpointing_enable()
    else:
        model.eval()
        model.requires_grad_(False)
    return model, tokenizer


def completion_start(offsets, prompt_length: int) -> int:
    """First token index whose character span ends past ``prompt_length``.

    The boundary is located by token *end* offset rather than *start*. A BPE
    tokenizer may merge the prompt's trailing whitespace with the completion's
    first word into one token; filtering on start offsets would then skip the
    few characters that straddle the boundary. ``end > boundary`` captures the
    straddling token too, which is the correct teacher-forced treatment.
    """

    for index, (start, end) in enumerate(offsets):
        if end > prompt_length:
            return index
    raise ValueError("completion tokenizes to zero tokens past the prompt boundary")


def encode_with_boundary(tokenizer, prompt: str, completion: str):
    """Encode ``prompt + completion`` once; return ``(input_ids, start_idx)``.

    ``start_idx`` is the first completion token in the joint encoding. Using the
    offset mapping of the joint encoding (rather than tokenizing the prompt on
    its own) is what makes the boundary correct under BPE cross-boundary merges.
    """

    try:
        enc = tokenizer(prompt + completion, return_tensors="pt", return_offsets_mapping=True)
    except (NotImplementedError, ValueError):
        enc = tokenizer(prompt + completion, return_tensors="pt")
    if "offset_mapping" in enc:
        offsets = enc["offset_mapping"][0].tolist()
        return enc["input_ids"], completion_start(offsets, len(prompt))
    # Tokenizers without offset_mapping (e.g. ChatGLM): tokenize the prompt
    # separately and use its length as the completion boundary.
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return enc["input_ids"], prompt_ids.shape[1]


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, completion: str) -> dict:
    """Total log-probability the policy assigns to ``completion`` after ``prompt``.

    This is the quantity Lemma 1 identifies as the connection coefficient: the
    probability that the policy would itself have emitted the certificate
    completion. Teacher forcing is used throughout, so the model never
    generates and any chat/thinking template is irrelevant.

    Returns the summed log-prob, the per-token mean, and the lengths in both
    tokens and characters. Tokens and characters are both recorded because
    token counts are not comparable across tokenizer families.
    """

    full_ids, boundary = encode_with_boundary(tokenizer, prompt, completion)
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)

    completion_length = full_ids.shape[1] - boundary
    if completion_length <= 0:
        raise ValueError("completion tokenizes to zero tokens after the prompt")

    logits = model(full_ids).logits.float()
    # Position t predicts token t+1; the completion span is [boundary, seq_len).
    log_probs = torch.log_softmax(logits[0, boundary - 1 : -1], dim=-1)
    targets = full_ids[0, boundary:]
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    total = float(token_log_probs.sum())
    return {
        "logprob_sum": total,
        "logprob_mean_per_token": total / completion_length,
        "n_tokens": int(completion_length),
        "n_chars": len(completion),
        "log10_c": total / np.log(10.0),
        "token_log_probs": [float(value) for value in token_log_probs],
    }


def fit_line(x: list[float], y: list[float]) -> dict:
    """Least-squares fit with R^2 and the slope's standard error.

    The paper's relation log10(c) = L * log10(p_bar) is an identity once
    p_bar is *defined* as the mean per-token probability; the empirical
    question is whether a single slope fits across lengths, which is what R^2
    answers. A poor fit is a finding, not a failed measurement, so the fit
    quality is returned alongside the slope rather than discarded.
    """

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    n = len(x_array)
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "slope_stderr": float("nan"), "n": n}

    slope, intercept = np.polyfit(x_array, y_array, 1)
    predicted = slope * x_array + intercept
    residual_ss = float(((y_array - predicted) ** 2).sum())
    total_ss = float(((y_array - y_array.mean()) ** 2).sum())
    r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")

    # Standard error of the slope under the usual homoscedastic assumption.
    variance_x = float(((x_array - x_array.mean()) ** 2).sum())
    stderr = ((residual_ss / (n - 2)) / variance_x) ** 0.5 if n > 2 and variance_x > 0 else float("nan")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "slope_stderr": float(stderr),
        "n": n,
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine between two gradient vectors, float32-accumulated in big chunks.

    The container has a ~120 GB cgroup memory cap, so the two float32 vectors
    (32 GB each for an 8B model) cannot both be materialized at once. Big
    chunks (64M elements) keep the loop short while bounding the working set."""
    a = a.flatten()
    b = b.flatten()
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    chunk = 1 << 26  # 64M elements ~ 256 MB float32
    for start in range(0, a.numel(), chunk):
        ac = a[start : start + chunk].float()
        bc = b[start : start + chunk].float()
        dot += float(torch.dot(ac, bc))
        norm_a += float(torch.dot(ac, ac))
        norm_b += float(torch.dot(bc, bc))
    denom = (norm_a * norm_b) ** 0.5
    return dot / denom if denom > 0 else float("nan")


def flat_grad(model) -> torch.Tensor:
    """Concatenate the current gradients into one vector, zeros where absent.

    Moves each parameter's gradient to CPU *before* concatenating so the GPU
    never holds model + per-param grads + the concatenated copy at once (that
    triple spikes to ~84 GB for a 14B model and OOMs an 80 GB GPU)."""

    pieces = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is not None:
                pieces.append(parameter.grad.detach().flatten().cpu())
            else:
                pieces.append(torch.zeros(parameter.numel(), device="cpu", dtype=parameter.dtype))
    model.zero_grad(set_to_none=True)  # free GPU grads immediately
    return torch.cat(pieces)


# --- Dataset loaders (shared by E1/E2c/E3/E5/E6). ---------------------------


def load_humaneval(limit: int | None = None) -> list[dict]:
    """HumanEval: prompt is the signature+docstring, certificate is the body.

    Items also carry the test-harness fields (``test``, ``entry_point``) that
    E1 needs for real test execution; other experiments ignore them.
    """
    from datasets import load_dataset

    rows = load_dataset("openai/openai_humaneval", split="test")
    items = [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "certificate": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in rows
    ]
    return items[:limit] if limit else items


def load_mbpp(limit: int | None = None) -> list[dict]:
    """MBPP (sanitized): prompt is the NL task + first test, certificate is code.

    The model writes the whole function from scratch. ``test_list`` are bare
    assert statements (they call the function directly) and ``test_imports`` are
    module imports run first, both used by E1's real-test execution.
    """
    from datasets import load_dataset

    rows = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    items = []
    for row in rows:
        tests = "\n".join(row["test_list"])
        prompt = f'"""\n{row["prompt"]}\n{tests}\n"""\n'
        items.append(
            {
                "task_id": str(row["task_id"]),
                "prompt": prompt,
                "certificate": row["code"],
                "test_list": row["test_list"],
                "test_imports": row["test_imports"],
            }
        )
    return items[:limit] if limit else items


def load_quixbugs(limit: int | None = None) -> list[dict]:
    """QuixBugs: single-line defects. Anchors the short end of the length axis.

    Reads the local JSONL of {task_id, prompt, certificate} prepared by
    prepare_quixbugs.py (certificate = the corrected line(s) only).
    """
    path = DATA_ROOT / "data" / "quixbugs.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run prepare_quixbugs.py first (see its docstring)"
        )
    items = [json.loads(line) for line in path.open()]
    return items[:limit] if limit else items


LOADERS = {"humaneval": load_humaneval, "mbpp": load_mbpp, "quixbugs": load_quixbugs}


def load_dataset_items(dataset: str, limit: int | None = None) -> list[dict]:
    """Load tasks for ``dataset`` (one of LOADERS), optionally capped at ``limit``."""
    return LOADERS[dataset](limit)


def run_mbpp_test_status(completion: str, test_list: list[str], test_imports, timeout: int = 30) -> str:
    """Run MBPP's asserts; return 'pass' | 'fail' | 'timeout' | 'harness_error'.

    MBPP has no entry-point wrapper: ``test_list`` are bare assert statements
    that call the function directly, and ``test_imports`` are module imports
    (a list of import strings) to run first. Same subprocess isolation + timeout
    as the HumanEval harness -- never ``exec`` untrusted model output in the
    main process.
    """
    import subprocess
    import sys

    imports = "\n".join(test_imports) if test_imports else ""
    program = imports + "\n" + completion + "\n" + "\n".join(test_list) + "\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=timeout,
        )
        return "pass" if result.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "harness_error"


def run_mbpp_test(completion: str, test_list: list[str], test_imports, timeout: int = 30) -> bool:
    """Bool wrapper (backward compat). True iff the asserts pass."""
    return run_mbpp_test_status(completion, test_list, test_imports, timeout) == "pass"

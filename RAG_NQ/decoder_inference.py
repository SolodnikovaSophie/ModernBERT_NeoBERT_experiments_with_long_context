"""HuggingFace causal LM wrapper tailored to oracle QA."""
import gc
import logging
import re
import time
import warnings
from typing import Optional, Tuple, Dict, Any
import torch
import transformers


# Greedy decoding still triggers `UserWarning` from transformers about
# unused sampling params even when we override `do_sample=False`. Silence
# just that family of warnings; everything else is left untouched.
warnings.filterwarnings(
    "ignore",
    message=r".*do_sample.* is set to `False`.*",
    category=UserWarning,
    module=r"transformers\.generation\..*",
)
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# transformers >=4.56 renamed `torch_dtype` to `dtype` and emits a deprecation
# warning when the old name is used. Pick the right kwarg at import time.
def _dtype_kwarg() -> str:
    try:
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    except Exception:
        return "torch_dtype"


_DTYPE_KW = _dtype_kwarg()


# Some 3rd-party model code (Nanbeige, MiniCPM4) calls torch.is_autocast_enabled
# with an argument; that raises on PyTorch 2.2+. Patch defensively.
_orig_is_autocast_enabled = torch.is_autocast_enabled


def _patched_is_autocast_enabled(*args, **kwargs):
    return _orig_is_autocast_enabled()


torch.is_autocast_enabled = _patched_is_autocast_enabled


def _select_dtype(preferred: str) -> torch.dtype:
    if preferred == "bfloat16":
        return (torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16)
    if preferred == "float16":
        return torch.float16
    if preferred == "float32":
        return torch.float32
    return torch.float16


class DecoderInferenceModel:
    """Loads a causal LM and produces extractive answers from a context+question prompt."""

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        quantize: bool = False,
        device: str = "cuda",
        use_chat_template: bool = True,
        qwen3_no_think: bool = False,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.use_chat_template = use_chat_template
        self.qwen3_no_think = qwen3_no_think
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.is_quantized = bool(quantize)
        self.last_prompt: Optional[str] = None

        force_cpu = (device == "cpu") or (not torch.cuda.is_available())
        self.device = torch.device("cpu" if force_cpu else "cuda")

        self.logger.info(f"[LOAD] tokenizer: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True)
        except Exception as e:
            self.logger.warning(f"fast tokenizer failed ({e}); using slow")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        mpe = (getattr(cfg, "max_position_embeddings", None)
               or getattr(cfg, "n_positions", None))
        if not mpe or mpe <= 0:
            mpe = getattr(self.tokenizer, "model_max_length", 2048)
            if not mpe or mpe >= int(1e9):
                mpe = 2048
        self.max_position_embeddings = int(mpe)
        self.logger.info(f"[LOAD] max_position_embeddings={self.max_position_embeddings}")

        torch_dtype = _select_dtype(dtype)
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True, _DTYPE_KW: torch_dtype}

        if quantize:
            if not HAS_BNB:
                raise RuntimeError("bitsandbytes is not installed but quantize=True")
            if force_cpu:
                raise RuntimeError("4-bit quantization requires CUDA")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(torch.bfloat16
                                        if torch_dtype == torch.bfloat16 else torch.float16),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb
            load_kwargs["device_map"] = "auto"

        self.logger.info(f"[LOAD] model: {model_path} | dtype={dtype} | quantize={quantize}")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not quantize:
            self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"[LOAD] loaded in {time.time()-t0:.1f}s")

        try:
            self._apply_generation_config()
        except Exception as e:
            self.logger.debug(f"generation_config merge skipped: {e}")

        # If user requested greedy decoding (temperature=0), purge sampling
        # params from model.generation_config — otherwise transformers spams a
        # warning on every .generate() call ("do_sample=False but temperature
        # is set to 0.6...").
        if self.temperature <= 0.0:
            gc_obj = getattr(self.model, "generation_config", None)
            if gc_obj is not None:
                gc_obj.do_sample = False
                gc_obj.temperature = 1.0
                gc_obj.top_p = 1.0
                gc_obj.top_k = 0

        self.is_qwen3 = "qwen3" in model_path.lower()
        self.is_deepseek_r1 = ("deepseek-r1" in model_path.lower()
                               or "r1-distill" in model_path.lower())
        self.is_phi = "phi" in model_path.lower()

        self.logger.info(f"[LOAD] params: {sum(p.numel() for p in self.model.parameters()):,}")

    # ----- generation config helpers -----

    def _apply_generation_config(self):
        gc_obj = getattr(self.model, "generation_config", None)
        if gc_obj is None:
            return
        # Use top_p, repetition_penalty from generation_config if present.
        # Temperature is taken from the user CLI (we want greedy by default).
        self.top_p = float(getattr(gc_obj, "top_p", 1.0) or 1.0)
        self.repetition_penalty = float(
            getattr(gc_obj, "repetition_penalty", 1.0) or 1.0)
        self.top_k = getattr(gc_obj, "top_k", None)
        if self.top_k is not None:
            self.top_k = int(self.top_k)

    def _special_token_kwargs(self) -> Dict[str, Any]:
        gc_obj = getattr(self.model, "generation_config", None)
        out: Dict[str, Any] = {}
        pad = (getattr(gc_obj, "pad_token_id", None)
               if gc_obj is not None else None) or self.tokenizer.pad_token_id
        eos = (getattr(gc_obj, "eos_token_id", None)
               if gc_obj is not None else None) or self.tokenizer.eos_token_id
        out["pad_token_id"] = pad
        out["eos_token_id"] = eos
        return out

    # ----- prompt building -----

    def build_prompt(self, system_prompt: str, user_template: str,
                     context: str, question: str) -> str:
        user_content = (user_template
                        .replace("{context}", context)
                        .replace("{question}", question))
        if (self.is_qwen3 and self.qwen3_no_think
                and not re.search(r"(?i)/no_think\s*$", user_content.strip())):
            user_content = user_content.rstrip() + "\n\n/no_think"
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})
            try:
                kw = dict(tokenize=False, add_generation_prompt=True)
                if self.is_qwen3:
                    kw["enable_thinking"] = False
                text = self.tokenizer.apply_chat_template(messages, **kw)
                text = re.sub(r"<think>\s*</think>\s*", "", text,
                              flags=re.IGNORECASE | re.DOTALL)
                return text
            except Exception as e:
                self.logger.debug(f"apply_chat_template failed: {e}")
        return ((system_prompt + "\n\n" if system_prompt else "")
                + user_content + "\nAnswer:")

    # ----- generation -----

    @torch.no_grad()
    def generate(self, prompt_text: str,
                 max_input_tokens: Optional[int] = None) -> Tuple[str, int]:
        """Returns (post-processed answer, input_token_count). Raises CUDA OOM
        on memory failure; caller decides how to recover."""
        self.last_prompt = prompt_text
        if max_input_tokens is None:
            max_input_tokens = self.max_position_embeddings - self.max_new_tokens - 8
        max_input_tokens = max(1, max_input_tokens)

        inputs = self.tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=max_input_tokens,
        )
        input_len = int(inputs.input_ids.shape[1])
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        do_sample = self.temperature > 0.0
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            **self._special_token_kwargs(),
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            top_p = getattr(self, "top_p", 1.0)
            if top_p and top_p > 0:
                gen_kwargs["top_p"] = top_p
            top_k = getattr(self, "top_k", None)
            if top_k:
                gen_kwargs["top_k"] = top_k

        try:
            outputs = self.model.generate(**inputs, **gen_kwargs)
        except (TypeError, AttributeError) as e:
            if "is_autocast_enabled" in str(e):
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            else:
                raise

        new_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._postprocess(text), input_len

    def _postprocess(self, text: str) -> str:
        # Strip any chain-of-thought leak.
        text = re.sub(r"<think>.*?</think>", "", text,
                      flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        # Cut "Answer:" prefix if echoed.
        text = re.sub(r"^\s*Answer:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ----- size + cleanup -----

    def get_model_size_bytes(self) -> int:
        # In 4-bit, .element_size()=1 (int8 storage); the result reflects on-GPU
        # storage of weights, which matches what nvidia-smi sees.
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def free(self):
        try:
            if not self.is_quantized:
                self.model.cpu()
        except Exception:
            pass
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

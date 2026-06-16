"""
Wrapper classes for Tenstorrent hardware models.

These lightweight wrappers store model metadata and provide the interface
expected by ComfyUI nodes. Actual inference runs on the tt-metal HTTP server
(see ``comfy/backends/tt_http_client.py``).
"""

import torch
import logging
import os
from typing import Dict, Any, List, Tuple, Optional
from .utils import get_model_config, format_bytes

logger = logging.getLogger(__name__)


class TTModelWrapper:
    """
    Lightweight wrapper for Tenstorrent UNet/DiT models.

    Carries the HTTP client + server metadata used by TT_KSampler. Implements
    ModelPatcher-compatible stubs so it survives ComfyUI's model management.
    """

    def __init__(self, client, model_type: str, server_info: Optional[Dict[str, Any]] = None, lora: Optional[Dict[str, Any]] = None):
        """
        Initialize model wrapper.

        Args:
            client: TTHttpClient instance
            model_type: Model type (sdxl, sd35, sd14)
            server_info: Optional dict from the server /health response
            lora: Optional {"lora_path": str, "lora_scale": float} attached via TT_LoraLoader
        """
        self.client = client
        self.model_type = model_type
        self.server_info = server_info or {}
        self.lora = lora
        # Stable identifier for display/info (the HTTP server serves one model).
        self.model_id = self.server_info.get("model", model_type)
        self.config = get_model_config(model_type)

        # ModelPatcher-compatible attributes for ComfyUI model management
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.model = None
        self.parent = None

        logger.info(f"Initialized TTModelWrapper for {model_type} (server: {self.model_id})")

    def model_size(self) -> int:
        """Estimate model size in bytes."""
        size_gb = self.config.get("model_size_gb", 7.0)
        return int(size_gb * 1024 * 1024 * 1024)

    # ModelPatcher-compatible stubs — prevent AttributeError if this wrapper
    # is accidentally passed through ComfyUI's standard model management pipeline.
    # The actual model lives on the bridge server, not in this process.

    def clone(self):
        return TTModelWrapper(self.client, self.model_type, self.server_info, self.lora)

    def with_lora(self, lora: Optional[Dict[str, Any]]):
        """Return a clone with LoRA metadata attached (used by TT_LoraLoader)."""
        return TTModelWrapper(self.client, self.model_type, self.server_info, lora)

    def is_clone(self, other):
        return isinstance(other, TTModelWrapper) and self.model_id == other.model_id

    def lowvram_patch_counter(self):
        return 0

    def model_dtype(self):
        return torch.float32

    def loaded_size(self):
        return 0

    def model_patches_to(self, *args, **kwargs):
        pass

    def detach(self, *args, **kwargs):
        pass

    def partially_unload(self, *args, **kwargs):
        return 0

    def partially_load(self, *args, **kwargs):
        return 0

    def current_loaded_device(self):
        return torch.device("cpu")

    def model_patches_models(self):
        return []

    def is_dynamic(self):
        return False

    def __repr__(self) -> str:
        return f"TTModelWrapper(model_id={self.model_id}, type={self.model_type})"


class TTCLIPWrapper:
    """
    Wrapper for Tenstorrent CLIP text encoders.

    Provides the CLIP interface expected by ComfyUI nodes (CLIPTextEncode).
    Tokenization is handled locally using the transformers library.
    Text encoding is deferred to the bridge server during inference.

    This wrapper enables TTCLIPWrapper to work with standard ComfyUI nodes
    like CLIPTextEncode, which calls clip.tokenize() and clip.encode_from_tokens_scheduled().
    """

    def __init__(self, client, model_type: str, server_info: Optional[Dict[str, Any]] = None):
        """
        Initialize CLIP wrapper.

        Args:
            client: TTHttpClient instance
            model_type: Model type (sdxl, sd35, sd14)
            server_info: Optional dict from the server /health response
        """
        self.client = client
        self.model_type = model_type
        self.server_info = server_info or {}
        self.model_id = self.server_info.get("model", model_type)
        self.config = get_model_config(model_type)

        # CLIP configuration
        self.layer_idx = None
        self.tokenizer_options = {}
        self.use_clip_schedule = False
        self.apply_hooks_to_conds = None

        # Token configuration (matches ComfyUI defaults)
        self.max_length = 77
        self.start_token = 49406
        self.end_token = 49407
        self.pad_token = 49407  # SDXL pads with end token for CLIP-L, 0 for CLIP-G

        # Initialize tokenizers
        self._init_tokenizers()

        logger.info(f"Initialized TTCLIPWrapper for {model_type} (ID: {self.model_id})")

    def _init_tokenizers(self):
        """
        Initialize local tokenizers for text processing.

        Uses the transformers CLIPTokenizer, matching ComfyUI's approach.
        For SDXL, both CLIP-L and CLIP-G tokenizers are loaded.
        """
        try:
            from transformers import CLIPTokenizer

            # Get the ComfyUI tokenizer path for compatibility
            comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sd1_tokenizer_path = os.path.join(comfy_path, "comfy", "sd1_tokenizer")

            if self.model_type == "sdxl":
                # SDXL uses both CLIP-L and CLIP-G tokenizers
                # CLIP-L: standard SD1 tokenizer
                if os.path.exists(sd1_tokenizer_path):
                    self.tokenizer_l = CLIPTokenizer.from_pretrained(sd1_tokenizer_path)
                else:
                    # Fallback to HuggingFace
                    self.tokenizer_l = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-large-patch14"
                    )
                # CLIP-G: uses same tokenizer with different configuration
                # (Both use same vocab, just different model architecture)
                self.tokenizer_g = self.tokenizer_l  # Same tokenizer, different model
                logger.info("Initialized SDXL tokenizers (CLIP-L and CLIP-G)")
            else:
                # SD1.x/SD2.x use single CLIP tokenizer
                if os.path.exists(sd1_tokenizer_path):
                    self.tokenizer = CLIPTokenizer.from_pretrained(sd1_tokenizer_path)
                else:
                    self.tokenizer = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-large-patch14"
                    )
                logger.info("Initialized SD1.x tokenizer")

        except ImportError as e:
            logger.error(f"Failed to import CLIPTokenizer: {e}")
            raise RuntimeError(
                "transformers library required for TTCLIPWrapper tokenization. "
                "Install with: pip install transformers"
            )

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs) -> Dict:
        """
        Tokenize text for CLIP encoding.

        This method is called by CLIPTextEncode node (nodes.py:73).
        Returns token weight pairs in the format expected by ComfyUI.

        Args:
            text: Input text to tokenize
            return_word_ids: Whether to include word IDs in output
            **kwargs: Additional tokenizer options

        Returns:
            Dictionary with tokenized text:
            - For SDXL: {"g": [...], "l": [...]} with both CLIP-G and CLIP-L tokens
            - For SD1.x: {"l": [...]} with CLIP-L tokens
        """
        tokenizer_options = kwargs.get("tokenizer_options", {})
        if len(self.tokenizer_options) > 0:
            tokenizer_options = {**self.tokenizer_options, **tokenizer_options}

        if self.model_type == "sdxl":
            # SDXL requires both CLIP-L and CLIP-G tokenization
            tokens_l = self._tokenize_single(
                text, self.tokenizer_l, return_word_ids,
                pad_with_end=True, **kwargs
            )
            tokens_g = self._tokenize_single(
                text, self.tokenizer_g, return_word_ids,
                pad_with_end=False, **kwargs  # CLIP-G pads with 0
            )
            return {"l": tokens_l, "g": tokens_g}
        else:
            # SD1.x/SD2.x single tokenizer
            tokens = self._tokenize_single(
                text, self.tokenizer, return_word_ids,
                pad_with_end=True, **kwargs
            )
            return {"l": tokens}

    def _tokenize_single(
        self,
        text: str,
        tokenizer,
        return_word_ids: bool = False,
        pad_with_end: bool = True,
        **kwargs
    ) -> List[List[Tuple]]:
        """
        Tokenize text with a single tokenizer.

        Implements tokenization matching ComfyUI's SDTokenizer.tokenize_with_weights().

        Args:
            text: Input text
            tokenizer: CLIPTokenizer instance
            return_word_ids: Include word IDs in output
            pad_with_end: Pad with end token (True) or 0 (False)

        Returns:
            List of batches, each batch is list of (token, weight) or (token, weight, word_id) tuples
        """
        # Handle empty text
        if not text or text.strip() == "":
            text = ""

        # Tokenize using transformers
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=True
        )

        token_ids = encoded["input_ids"]

        # Build token weight pairs
        # Format: [(token_id, weight, word_id), ...] or [(token_id, weight), ...]
        batch = []

        for i, token_id in enumerate(token_ids):
            if return_word_ids:
                # word_id 0 is reserved for special tokens
                word_id = 0 if i == 0 or i == len(token_ids) - 1 else i
                batch.append((token_id, 1.0, word_id))
            else:
                batch.append((token_id, 1.0))

        # Pad to max_length
        pad_token = self.end_token if pad_with_end else 0
        while len(batch) < self.max_length:
            if return_word_ids:
                batch.append((pad_token, 1.0, 0))
            else:
                batch.append((pad_token, 1.0))

        # Return as single batch (ComfyUI format)
        return [batch]

    def encode_from_tokens_scheduled(
        self,
        tokens: Dict,
        unprojected: bool = False,
        add_dict: Dict = {},
        show_pbar: bool = True
    ) -> List[Tuple[torch.Tensor, Dict]]:
        """
        Encode tokens to conditioning format.

        This method is called by CLIPTextEncode node (nodes.py:74).
        For Tenstorrent, we create placeholder embeddings and store the
        original prompt text - actual encoding happens on the bridge server.

        Args:
            tokens: Tokenized text from tokenize()
            unprojected: Whether to return unprojected embeddings
            add_dict: Additional metadata to include
            show_pbar: Whether to show progress bar (ignored)

        Returns:
            List of (conditioning_tensor, metadata_dict) tuples in CONDITIONING format
        """
        # Extract original text from tokens for bridge server
        # We need to reconstruct or pass through the original prompt
        original_text = self._reconstruct_text_from_tokens(tokens)

        # Create placeholder conditioning tensor
        # Shape depends on model type
        if self.model_type == "sdxl":
            # SDXL: concatenated CLIP-L (768) + CLIP-G (1280) = 2048 dim
            # Sequence length 77
            cond_shape = (1, 77, 2048)
            pooled_shape = (1, 1280)  # CLIP-G pooled output
        elif self.model_type == "sd35":
            # SD3.5: different architecture
            cond_shape = (1, 77, 4096)
            pooled_shape = (1, 4096)
        else:
            # SD1.x: CLIP-L only, 768 dim
            cond_shape = (1, 77, 768)
            pooled_shape = (1, 768)

        # Create placeholder tensors
        # These will be replaced by actual embeddings from bridge server
        cond = torch.zeros(cond_shape, dtype=torch.float32)
        pooled = torch.zeros(pooled_shape, dtype=torch.float32)

        # Build metadata dictionary
        # The 'prompt' key is used by TT_Denoise to extract text for bridge server
        pooled_dict = {
            "pooled_output": pooled,
            "prompt": original_text,
            "tt_clip_wrapper": True,  # Flag indicating this is from TT wrapper
            "model_id": self.model_id,
        }

        # Add any extra metadata
        pooled_dict.update(add_dict)

        # Add hooks if present
        if self.apply_hooks_to_conds:
            pooled_dict["hooks"] = self.apply_hooks_to_conds

        return [(cond, pooled_dict)]

    def encode_from_tokens(
        self,
        tokens: Dict,
        return_pooled: bool = False,
        return_dict: bool = False
    ):
        """
        Basic token encoding interface.

        For Tenstorrent wrapper, this creates placeholder embeddings.
        Actual encoding is deferred to bridge server.

        Args:
            tokens: Tokenized text
            return_pooled: Whether to return pooled output
            return_dict: Whether to return as dictionary

        Returns:
            Conditioning tensor, optionally with pooled output
        """
        # Get scheduled encoding
        scheduled = self.encode_from_tokens_scheduled(tokens)
        cond, pooled_dict = scheduled[0]
        pooled = pooled_dict.get("pooled_output", None)

        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            out["prompt"] = pooled_dict.get("prompt", "")
            if self.apply_hooks_to_conds:
                out["hooks"] = self.apply_hooks_to_conds
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text: str) -> torch.Tensor:
        """
        High-level encoding method.

        Tokenizes and encodes text in one call.

        Args:
            text: Input text to encode

        Returns:
            Conditioning tensor
        """
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def _reconstruct_text_from_tokens(self, tokens: Dict) -> str:
        """
        Reconstruct original text from tokens.

        Since we tokenize locally, we need to decode back to text
        for the bridge server. This is a best-effort reconstruction.

        Args:
            tokens: Tokenized text dictionary

        Returns:
            Reconstructed text string
        """
        try:
            # Get token IDs from the first available key
            if "l" in tokens:
                token_batch = tokens["l"]
            elif "g" in tokens:
                token_batch = tokens["g"]
            else:
                return ""

            if not token_batch:
                return ""

            # Extract token IDs from (token, weight) or (token, weight, word_id) tuples
            token_ids = []
            for item in token_batch[0]:  # First batch
                if isinstance(item, (list, tuple)):
                    token_id = item[0]
                else:
                    token_id = item

                # Skip special tokens and padding
                if token_id in [self.start_token, self.end_token, 0]:
                    continue
                if isinstance(token_id, int):
                    token_ids.append(token_id)

            if not token_ids:
                return ""

            # Decode using tokenizer
            if self.model_type == "sdxl":
                text = self.tokenizer_l.decode(token_ids, skip_special_tokens=True)
            else:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            return text.strip()

        except Exception as e:
            logger.warning(f"Failed to reconstruct text from tokens: {e}")
            return ""

    def clip_layer(self, layer_idx: int) -> None:
        """
        Set which transformer layer output to use.

        Args:
            layer_idx: Layer index (-1 for last, -2 for penultimate, etc.)
        """
        self.layer_idx = layer_idx

    def set_tokenizer_option(self, option_name: str, value: Any) -> None:
        """
        Set a tokenizer configuration option.

        Args:
            option_name: Name of the option
            value: Option value
        """
        self.tokenizer_options[option_name] = value

    def clone(self) -> "TTCLIPWrapper":
        """
        Create a clone of this wrapper.

        Returns:
            New TTCLIPWrapper instance with same configuration
        """
        cloned = TTCLIPWrapper(self.client, self.model_type, self.server_info)
        cloned.layer_idx = self.layer_idx
        cloned.tokenizer_options = self.tokenizer_options.copy()
        cloned.use_clip_schedule = self.use_clip_schedule
        cloned.apply_hooks_to_conds = self.apply_hooks_to_conds
        return cloned

    def load_model(self):
        """
        Load model to device.

        For Tenstorrent wrapper, this is a no-op since the model
        is already loaded on the bridge server.

        Returns:
            self for compatibility
        """
        # Model is loaded on bridge server, nothing to do here
        return self

    def get_key_patches(self):
        """
        Get key patches for LoRA compatibility.

        Returns:
            Empty dictionary (LoRA not supported on bridge)
        """
        return {}

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        """
        Add patches for LoRA compatibility.

        For Tenstorrent wrapper, LoRA is not supported.

        Returns:
            Empty tuple
        """
        logger.warning("LoRA patches not supported on Tenstorrent CLIP wrapper")
        return ()

    def __repr__(self) -> str:
        return f"TTCLIPWrapper(model_id={self.model_id}, type={self.model_type})"


class TTVAEWrapper:
    """
    Lightweight wrapper for Tenstorrent VAE encoder/decoder.

    Carries the HTTP client + server metadata. VAE encode/decode are dispatched
    to the tt-metal server's ``/vae/encode`` and ``/vae/decode`` endpoints.
    """

    def __init__(self, client, model_type: str, server_info: Optional[Dict[str, Any]] = None):
        """
        Initialize VAE wrapper.

        Args:
            client: TTHttpClient instance
            model_type: Model type (sdxl, sd35, sd14)
            server_info: Optional dict from the server /health response
        """
        self.client = client
        self.model_type = model_type
        self.server_info = server_info or {}
        self.model_id = self.server_info.get("model", model_type)
        self.config = get_model_config(model_type)

        logger.info(f"Initialized TTVAEWrapper for {model_type} (server: {self.model_id})")

    def __repr__(self) -> str:
        return f"TTVAEWrapper(model_id={self.model_id}, type={self.model_type})"

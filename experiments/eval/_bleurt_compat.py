"""Compatibility shim for `bleurt-pytorch` against transformers >= 5.0.

bleurt-pytorch was last released against transformers 4.x and imports three
symbols that no longer live in the same place in t5.x. Import this module
BEFORE `from bleurt_pytorch import ...` to patch them in-place.
"""
import sys
import types

import transformers.pytorch_utils as _pu
import transformers.models.bert as _bert_mod
from transformers import BertTokenizerFast, PreTrainedModel


# 1) `find_pruneable_heads_and_indices` was removed (head pruning deprecated).
#    BLEURT never prunes heads at inference, so a no-op stub is enough.
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    _pu.find_pruneable_heads_and_indices = lambda *_a, **_kw: (set(), None)

# 2) The `transformers.models.bert.tokenization_bert_fast` submodule was
#    removed; `BertTokenizerFast` is now only exported from the top-level
#    `transformers` namespace.
_mod_name = "transformers.models.bert.tokenization_bert_fast"
if _mod_name not in sys.modules:
    _fake = types.ModuleType(_mod_name)
    _fake.BertTokenizerFast = BertTokenizerFast
    sys.modules[_mod_name] = _fake
    _bert_mod.tokenization_bert_fast = _fake

# 3) `PreTrainedModel.get_head_mask` was removed. The BLEURT BERT call passes
#    `head_mask=None`, so a passthrough returning `[None] * num_layers` is
#    sufficient.
if not hasattr(PreTrainedModel, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        return head_mask
    PreTrainedModel.get_head_mask = _get_head_mask

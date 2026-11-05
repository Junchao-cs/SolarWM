"""Compose the frozen official Stage2 and Memory adapters for inference."""

import hashlib
from collections import OrderedDict
from copy import deepcopy

from .lora import H3LoRARuntime


def attach_memory_adapter(wrapped, frozen):
    import torch

    digest = hashlib.sha256()
    for key, value in frozen.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().view(torch.uint8).numpy().tobytes())
    frozen_sha = digest.hexdigest()
    wrapped.add_adapter("memory", deepcopy(frozen.peft_config))
    wrapped.base_model.set_adapter(["default", "memory"])
    memory_parameter_ids = set()
    for name, parameter in wrapped.named_parameters():
        if ".memory." in name and (".lora_A." in name or ".lora_B." in name):
            parameter.data = parameter.data.to(torch.bfloat16)
            memory_parameter_ids.add(id(parameter))
    wrapped.requires_grad_(False)
    standard = frozen.peft_module.get_peft_model_state_dict(
        wrapped,
        state_dict=wrapped.state_dict(keep_vars=True),
        adapter_name="memory",
        save_embedding_layers=False,
    )
    parameters = OrderedDict((k, standard[k]) for k in sorted(standard))
    if set(parameters) != set(frozen.parameter_by_key):
        raise ValueError("Memory and frozen LoRA target keys differ")
    if {id(p) for p in parameters.values()} != memory_parameter_ids:
        raise ValueError("Memory adapter parameter identity differs")
    if any(p.requires_grad for p in frozen.parameters):
        raise ValueError("Released adapter must stay frozen")
    if any(torch.count_nonzero(p).item() for k, p in parameters.items() if ".lora_B." in k):
        raise ValueError("New Memory LoRA must initially contribute zero")
    new_identity = dict(
        frozen.base_identity,
        frozen_adapter={
            "sha256": frozen_sha,
            "parameter_count": frozen.parameter_count,
            "active_adapters": ["default", "memory"],
        },
    )
    memory = H3LoRARuntime(
        wrapped,
        frozen.targets,
        parameters,
        wrapped.peft_config["memory"],
        frozen.peft_module,
        new_identity,
        frozen.rank,
        frozen.alpha,
    )
    if memory.parameter_count != frozen.parameter_count:
        raise ValueError("Memory LoRA parameter count differs from released LoRA")
    return wrapped, memory, frozen

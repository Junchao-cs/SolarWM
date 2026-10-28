"""Borrowed tensor references and mutation stamps for private compiler inputs."""

from dataclasses import fields, is_dataclass
import torch


def tensor_stamp(t):
    return (id(t),t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device))

def tensor_references(value):
    result={};seen=set()
    def visit(v):
        if id(v) in seen:return
        seen.add(id(v))
        if isinstance(v,torch.Tensor):result[id(v)]=v
        elif is_dataclass(v):
            for f in fields(v):visit(getattr(v,f.name))
        elif isinstance(v,dict):
            for item in v.values():visit(item)
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
    visit(value)
    return tuple(result.values())

import ast
import sys
import json
import os.path as osp
from importlib import import_module
from easydict import EasyDict
import types


class Config:
    """config
    It supports .py format, allow access config values as attributes.
    The implementation refers to mmcv. 
    ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    """

    def __init__(self, cfg_dict):
        # self._cfg_dict = cfg_dict
        super().__setattr__('_cfg_dict', cfg_dict)
    
    @staticmethod
    def frompy(filename):
        filename = osp.abspath(osp.expanduser(filename))
        Config._check_py_syntax(filename)
        module_dir = osp.dirname(filename)
        module_name = osp.splitext(osp.basename(filename))[0]
        sys.path.insert(0, module_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = EasyDict({
            k: v for k, v in mod.__dict__.items()
            if not k.startswith('__')
            and not isinstance(v, types.ModuleType)
            and not isinstance(v, types.FunctionType)
        })
        del sys.modules[module_name]

        return Config(cfg_dict)

    @staticmethod
    def _check_py_syntax(filename):
        with open(filename, encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f'syntax error in <<<{filename}>>>: {e}')

    def __getattr__(self, key):
        return getattr(self._cfg_dict, key)

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = EasyDict(value)
        self._cfg_dict.__setattr__(key, value)

    def petty_print(self):
        print(json.dumps(self._cfg_dict, indent=4))

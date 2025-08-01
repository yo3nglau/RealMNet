# Copyright (c) OpenMMLab. All rights reserved.
import io
import logging
import os
import os.path as osp
import pkgutil
import re
from collections import OrderedDict, namedtuple
from importlib import import_module
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Optional

import torch

import mmengine
from mmengine.dist import get_dist_info
from mmengine.fileio import FileClient, get_file_backend
from mmengine.fileio import load as load_file
from mmengine.logging import print_log
from mmengine.model import BaseTTAModel, is_model_wrapper
from mmengine.utils import (apply_to, deprecated_function, digit_version,
                            mkdir_or_exist)
from mmengine.utils.dl_utils import load_url

# `MMENGINE_HOME` is the highest priority directory to save checkpoints
# downloaded from Internet. If it is not set, as a workaround, using
# `XDG_CACHE_HOME`` or `~/.cache` instead.
# Note that `XDG_CACHE_HOME` defines the base directory relative to which
# user-specific non-essential data files should be stored. If `XDG_CACHE_HOME`
# is either not set or empty, a default equal to `~/.cache` should be used.
ENV_MMENGINE_HOME = 'MMENGINE_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


class _IncompatibleKeys(
        namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__


def _get_mmengine_home():
    mmengine_home = os.path.expanduser(
        os.getenv(
            ENV_MMENGINE_HOME,
            os.path.join(
                os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmengine')))

    mkdir_or_exist(mmengine_home)
    return mmengine_home


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    missing_keys = []
    err_msg = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, local_state_dict, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_model_wrapper(module) or isinstance(module, BaseTTAModel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(local_state_dict, prefix, local_metadata,
                                     True, missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, '_load_state_dict_post_hooks'):
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    'Hooks registered with '
                    '``register_load_state_dict_post_hook`` are not expected '
                    'to return new values, if incompatible_keys need to be '
                    'modified, it should be done inplace.')

    load(module, state_dict)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print_log(err_msg, logger=logger, level=logging.WARNING)


def get_torchvision_models():
    import torchvision
    if digit_version(torchvision.__version__) < digit_version('0.13.0a0'):
        model_urls = dict()
        # When the version of torchvision is lower than 0.13, the model url is
        # not declared in `torchvision.model.__init__.py`, so we need to
        # iterate through `torchvision.models.__path__` to get the url for each
        # model.
        for _, name, ispkg in pkgutil.walk_packages(
                torchvision.models.__path__):
            if ispkg:
                continue
            _zoo = import_module(f'torchvision.models.{name}')
            if hasattr(_zoo, 'model_urls'):
                _urls = getattr(_zoo, 'model_urls')
                model_urls.update(_urls)
    else:
        # Since torchvision bumps to v0.13, the weight loading logic,
        # model keys and model urls have been changed. Here the URLs of old
        # version is loaded to avoid breaking back compatibility. If the
        # torchvision version>=0.13.0, new URLs will be added. Users can get
        # the resnet50 checkpoint by setting 'resnet50.imagent1k_v1',
        # 'resnet50' or 'ResNet50_Weights.IMAGENET1K_V1' in the config.
        json_path = osp.join(mmengine.__path__[0], 'hub/torchvision_0.12.json')
        model_urls = mmengine.load(json_path)
        if digit_version(torchvision.__version__) < digit_version('0.14.0a0'):
            weights_list = [
                cls for cls_name, cls in torchvision.models.__dict__.items()
                if cls_name.endswith('_Weights')
            ]
        else:
            weights_list = [
                torchvision.models.get_model_weights(model)
                for model in torchvision.models.list_models(torchvision.models)
            ]

        for cls in weights_list:
            # The name of torchvision model weights classes ends with
            # `_Weights` such as `ResNet18_Weights`. However, some model weight
            # classes, such as `MNASNet0_75_Weights` does not have any urls in
            # torchvision 0.13.0 and cannot be iterated. Here we simply check
            # `DEFAULT` attribute to ensure the class is not empty.
            if not hasattr(cls, 'DEFAULT'):
                continue
            # Since `cls.DEFAULT` can not be accessed by iterating cls, we set
            # default urls explicitly.
            cls_name = cls.__name__
            cls_key = cls_name.replace('_Weights', '').lower()
            model_urls[f'{cls_key}.default'] = cls.DEFAULT.url
            for weight_enum in cls:
                cls_key = cls_name.replace('_Weights', '').lower()
                cls_key = f'{cls_key}.{weight_enum.name.lower()}'
                model_urls[cls_key] = weight_enum.url

    return model_urls


def get_external_models():
    mmengine_home = _get_mmengine_home()
    default_json_path = osp.join(mmengine.__path__[0], 'hub/openmmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmengine_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmengine.__path__[0], 'hub/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmengine.__path__[0], 'hub/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def _process_mmcls_checkpoint(checkpoint):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Some checkpoints converted from 3rd-party repo don't
        # have the "state_dict" key.
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: Dict[str, Callable] = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger='current'):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to None
            logger (str): The logger for message. Defaults to 'current'.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        # print_log(
        #     f'Loads checkpoint by {class_name[10:]} backend from path: '
        #     f'{filename}',
        #     logger=logger)
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(filename, map_location):
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('http://', 'https://'))
def load_from_http(filename,
                   map_location=None,
                   model_dir=None,
                   progress=os.isatty(0)):
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (string, optional): directory in which to save the object,
            Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(
            filename,
            model_dir=model_dir,
            map_location=map_location,
            progress=progress)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_url(
                filename,
                model_dir=model_dir,
                map_location=map_location,
                progress=progress)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='pavi://')
def load_from_pavi(filename, map_location=None):
    """load checkpoint through the file path prefixed with pavi. In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with pavi prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    assert filename.startswith('pavi://'), \
        f'Expected filename startswith `pavi://`, but get {filename}'
    model_path = filename[7:]

    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')

    model = modelcloud.get(model_path)
    with TemporaryDirectory() as tmp_dir:
        downloaded_file = osp.join(tmp_dir, model.name)
        model.download(downloaded_file)
        checkpoint = torch.load(downloaded_file, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(
    prefixes=[r'(\S+\:)?s3://', r'(\S+\:)?petrel://'])
def load_from_ceph(filename, map_location=None, backend='petrel'):
    """load checkpoint through the file path prefixed with s3.  In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with s3 prefix
        map_location (str, optional): Same as :func:`torch.load`.
        backend (str, optional): The storage backend type.
            Defaults to 'petrel'.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    file_backend = get_file_backend(
        filename, backend_args={'backend': backend})
    with io.BytesIO(file_backend.get(filename)) as buffer:
        checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('modelzoo://', 'torchvision://'))
def load_from_torchvision(filename, map_location=None):
    """load checkpoint through the file path prefixed with modelzoo or
    torchvision.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    if filename.startswith('modelzoo://'):
        print_log(
            'The URL scheme of "modelzoo://" is deprecated, please '
            'use "torchvision://" instead',
            logger='current',
            level=logging.WARNING)
        model_name = filename[11:]
    else:
        model_name = filename[14:]
    return load_from_http(model_urls[model_name], map_location=map_location)


@CheckpointLoader.register_scheme(prefixes=('open-mmlab://', 'openmmlab://'))
def load_from_openmmlab(filename, map_location=None):
    """load checkpoint through the file path prefixed with open-mmlab or
    openmmlab.

    Args:
        filename (str): checkpoint file path with open-mmlab or
        openmmlab prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_external_models()
    prefix_str = 'open-mmlab://'
    if filename.startswith(prefix_str):
        model_name = filename[13:]
    else:
        model_name = filename[12:]
        prefix_str = 'openmmlab://'

    deprecated_urls = get_deprecated_model_names()
    if model_name in deprecated_urls:
        print_log(
            f'{prefix_str}{model_name} is deprecated in favor '
            f'of {prefix_str}{deprecated_urls[model_name]}',
            logger='current',
            level=logging.WARNING)
        model_name = deprecated_urls[model_name]
    model_url = model_urls[model_name]
    # check if is url
    if model_url.startswith(('http://', 'https://')):
        checkpoint = load_from_http(model_url, map_location=map_location)
    else:
        filename = osp.join(_get_mmengine_home(), model_url)
        if not osp.isfile(filename):
            raise FileNotFoundError(f'{filename} can not be found.')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='mmcls://')
def load_from_mmcls(filename, map_location=None):
    """load checkpoint through the file path prefixed with mmcls.

    Args:
        filename (str): checkpoint file path with mmcls prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_mmcls_models()
    model_name = filename[8:]
    checkpoint = load_from_http(
        model_urls[model_name], map_location=map_location)
    checkpoint = _process_mmcls_checkpoint(checkpoint)
    return checkpoint


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Defaults to None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
        OrderedDict storing model weights or a dict containing other
        information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def _load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict


def _load_checkpoint_to_model(model,
                              checkpoint,
                              strict=False,
                              logger=None,
                              revise_keys=[(r'^module\.', '')]):

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    return _load_checkpoint_to_model(model, checkpoint, strict, logger,
                                     revise_keys)


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    # stash metadata to put in state_dict later
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    state_dict = apply_to(state_dict, lambda x: hasattr(x, 'cpu'),
                          lambda x: x.cpu())
    state_dict._metadata = metadata
    return state_dict


@deprecated_function(
    since='0.3.0',
    removed_in='0.5.0',
    instructions='`_save_to_state_dict` will be deprecated in the future, '
    'please use `nn.Module._save_to_state_dict` directly.')
def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
        keep_vars (bool): Whether to keep the variable property of the
            parameters.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        if buf is not None and name not in module._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Defaults to False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_model_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    module._save_to_state_dict(destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(checkpoint,
                    filename,
                    file_client_args=None,
                    backend_args=None):
    """Save checkpoint to file.

    Args:
        checkpoint (dict): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            `backend_args` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.
    """
    if file_client_args is not None:
        print_log(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            logger='current',
            level=logging.WARNING)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set '
                'at the same time.')

    if filename.startswith('pavi://'):
        if file_client_args is not None or backend_args is not None:
            raise ValueError(
                '"file_client_args" or "backend_args" should be "None" if '
                'filename starts with "pavi://"')
        try:
            from pavi import exception, modelcloud
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except exception.NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        file_client = FileClient.infer_client(file_client_args, filename)
        if file_client_args is None:
            file_backend = get_file_backend(
                filename, backend_args=backend_args)
        else:
            file_backend = file_client

        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            file_backend.put(f.getvalue(), filename)


def find_latest_checkpoint(path: str) -> Optional[str]:
    """Find the latest checkpoint from the given path.

    Refer to https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/checkpoint.py  # noqa: E501

    Args:
        path(str): The path to find checkpoints.

    Returns:
        str or None: File path of the latest checkpoint.
    """
    save_file = osp.join(path, 'last_checkpoint')
    last_saved: Optional[str]
    if os.path.exists(save_file):
        with open(save_file) as f:
            last_saved = f.read().strip()
    else:
        print_log('Did not find last_checkpoint to be resumed.')
        last_saved = None
    return last_saved

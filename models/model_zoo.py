from .fcn import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'fcn32s_vgg16': fcn32s_vgg16,
    'fcn16s_vgg16': fcn16s_vgg16,
    'fcn8s_vgg16': fcn8s_vgg16
}

def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return _models.keys()
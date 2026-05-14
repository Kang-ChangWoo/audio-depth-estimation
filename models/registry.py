"""Model builder registry.

Register a builder with @register_builder("name"); build_model() dispatches
on cfg.model.name. Builders receive (cfg, gpu_ids) and return the model.
Adding a model = adding one decorated builder function — no if/elif edits.
"""

MODEL_BUILDERS = {}


def register_builder(name):
    def _wrap(fn):
        if name in MODEL_BUILDERS:
            raise ValueError(f"duplicate model builder: {name}")
        MODEL_BUILDERS[name] = fn
        return fn
    return _wrap


def build_model_from_registry(cfg, gpu_ids):
    name = getattr(cfg.model, 'name', 'unet_baseline')
    if name not in MODEL_BUILDERS:
        raise KeyError(
            f"unknown model.name '{name}'. "
            f"registered: {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[name](cfg, gpu_ids)

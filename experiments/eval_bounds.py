from argparse import ArgumentParser
from fastargs import get_current_config
from sublora.sublora_pipeline import SubLoRA


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description="SubLoRA GPT-2 bound evaluation")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()
    return config

if __name__ == "__main__":
    yaml_config = make_config()
    yaml_config = {key[1]: value for key, value in yaml_config.content.items()}
    yaml_config["action"] = "eval_bounds"
    # If a checkpoint path is provided, prefer any allocation metadata saved
    # inside the checkpoint so evaluation constructs the model identically
    # to training. This preserves learned allocations and any explicit caps.
    ckpt_path = yaml_config.get('model', {}).get('best_checkpoint_path') or yaml_config.get('model.best_checkpoint_path')
    if ckpt_path:
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            alloc = ckpt.get('allocation_config', None)
            if alloc is None:
                # try inside saved training config
                cfg = ckpt.get('config', {})
                alloc = cfg.get('sublora', {}).get('allocation_config', None)
            if alloc is not None:
                yaml_config.setdefault('sublora', {})
                yaml_config['sublora']['allocation_config'] = alloc
                print(f"Using allocation_config from checkpoint: {alloc}")
        except Exception:
            pass

    method = SubLoRA(yaml_config)
    method.get_bounds()
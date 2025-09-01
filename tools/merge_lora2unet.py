import argparse
import torch
import os
import shutil
from diffusers.models import UNet2DConditionModel
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import save_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--lora_ckpt_path", type=str)
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="root path of the sd1.5 model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="args.unet_ckpt_path + a new subfolder name",
        default="fused_lora",
    )
    parser.add_argument(
        "--unet_config_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5/unet/config.json",
        help="path to unet config, in the `unet` subfolder of args.unet_ckpt_path",
    )
    parser.add_argument(
        "--lora_keys", nargs="*", type=str, default=["to_q", "to_k", "to_v", "to_out"]
    )
    parser.add_argument("--negative_lora_keys", type=str, default="bias")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(
        "/hpc2hdd/home/hongfeizhang/hf_cache/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        subfolder="unet",
        force_download=True,
        timeout=120,
    )
    fused_state_dict = unet.state_dict()
    args.lora_ckpt_path = (
        "/hpc2hdd/home/hongfeizhang/hongfei_workspace/CameraCtrl/v3_sd15_adapter.ckpt"
    )
    print(f"Loading the lora weights from {args.lora_ckpt_path}")
    lora_state_dict = torch.load(args.lora_ckpt_path, map_location="cpu")
    if "state_dict" in lora_state_dict:
        lora_state_dict = lora_state_dict["state_dict"]
    print(f"Loading done")
    print(f"Fusing the lora weight to unet weight")
    used_lora_key = []
    for lora_key in args.lora_keys:
        unet_keys = [
            x
            for x in fused_state_dict.keys()
            if lora_key in x and args.negative_lora_keys not in x
        ]
        print(f"There are {len(unet_keys)} unet keys for lora key: {lora_key}")
        for unet_key in unet_keys:
            prefixes = unet_key.split(".")
            idx = prefixes.index(lora_key)
            lora_down_key = (
                ".".join(prefixes[:idx])
                + f".processor.{lora_key}_lora.down"
                + f".{prefixes[-1]}"
            )
            lora_up_key = (
                ".".join(prefixes[:idx])
                + f".processor.{lora_key}_lora.up"
                + f".{prefixes[-1]}"
            )
            assert lora_down_key in lora_state_dict and lora_up_key in lora_state_dict
            print(f"Fusing lora weight for {unet_key}")
            fused_state_dict[unet_key] = (
                fused_state_dict[unet_key]
                + torch.bmm(
                    lora_state_dict[lora_up_key][None, ...],
                    lora_state_dict[lora_down_key][None, ...],
                )[0]
                * args.lora_scale
            )
            used_lora_key.append(lora_down_key)
            used_lora_key.append(lora_up_key)
    assert len(set(used_lora_key) - set(lora_state_dict.keys())) == 0
    print(f"Fusing done")
    save_path = os.path.join(args.save_path, SAFETENSORS_WEIGHTS_NAME)
    print(f"Saving the fused state dict to {save_path}")
    save_file(fused_state_dict, save_path)
    config_dst_path = os.path.join(args.save_path, "config.json")
    print(f"Copying the unet config to {config_dst_path}")
    shutil.copy(args.unet_config_path, config_dst_path)
    print("Done!")

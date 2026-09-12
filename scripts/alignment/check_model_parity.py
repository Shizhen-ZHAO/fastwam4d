"""三个1d模型的实际VAE/loss/梯度/多步AdamW对照；不下载权重，不模拟ZeRO。"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types

import torch
import torch.utils.checkpoint


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-repo', required=True, type=Path)
    parser.add_argument('--target-repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('--steps must be >= 2 to check optimizer trajectories')
    roots = {'candidate': args.target_repo.resolve(), 'reference': args.reference_repo.resolve()}
    sys.path.insert(0, str(roots['candidate'] / 'src'))
    package('fastwam.utils', roots['candidate'] / 'src/fastwam/utils')
    mods = {}
    for name, repo in roots.items():
        alias = '_parity_' + name
        package(alias, repo / 'src/fastwam/models/wan22')
        mods[name] = {key: importlib.import_module(alias + '.' + key) for key in [
            'wan_video_dit', 'action_dit', 'mot', 'wan_video_vae', 'fastwam', 'fastwam_joint', 'fastwam_idm',
        ]}
    torch.set_num_threads(1)
    vcfg = dict(hidden_dim=24, in_dim=4, ffn_dim=48, out_dim=4, text_dim=20,
                freq_dim=12, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
                attn_head_dim=12, num_layers=2, has_image_input=False,
                seperated_timestep=True, fuse_vae_embedding_in_latents=True,
                action_conditioned=False, video_attention_mask_mode='first_frame_causal')
    acfg = dict(hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=20,
                freq_dim=12, eps=1e-6, num_heads=2, attn_head_dim=12, num_layers=2,
                action_rope_mode='1d')

    def model(name, variant, cond_prob):
        m = mods[name]
        video = m['wan_video_dit'].WanVideoDiT(**vcfg, use_gradient_checkpointing=False)
        action = m['action_dit'].ActionDiT(**acfg, use_gradient_checkpointing=False)
        mot = m['mot'].MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=False)
        vae = m['wan_video_vae'].WanVideoVAE38.__new__(m['wan_video_vae'].WanVideoVAE38)
        torch.nn.Module.__init__(vae)
        vae.model = m['wan_video_vae'].VideoVAE38_(dim=8, z_dim=4, dec_dim=8, num_res_blocks=1)
        vae.upsampling_factor, vae.temporal_downsample_factor, vae.z_dim = 16, 4, 4
        mean, std = torch.tensor([.11, -.2, .07, .3]), torch.tensor([.9, 1.2, .8, 1.4])
        if name == 'candidate':
            vae.register_buffer('mean', mean, persistent=False)
            vae.register_buffer('inv_std', std.reciprocal(), persistent=False)
        else:
            vae.mean, vae.std, vae.scale = mean, std, [mean, std.reciprocal()]
        module_name, class_name = {'uncond': ('fastwam', 'FastWAM'),
                                   'joint': ('fastwam_joint', 'FastWAMJoint'),
                                   'idm': ('fastwam_idm', 'FastWAMIDM')}[variant]
        net = getattr(m[module_name], class_name)(
            video, action, mot, vae, text_dim=20, proprio_dim=8,
            video_train_shift=5.0, video_infer_shift=5.0,
            action_train_shift=5.0, action_infer_shift=5.0,
            device='cpu', torch_dtype=torch.float32,
        )
        if variant == 'idm':
            net.video_cond_noise_prob = cond_prob
        net.eval().requires_grad_(False)
        net.dit.train().requires_grad_(True)
        net.proprio_encoder.train().requires_grad_(True)
        return net

    def compare_named(left, right, gradients=False):
        a, b = dict(left.named_parameters()), dict(right.named_parameters())
        assert a.keys() == b.keys()
        count = 0
        for key in a:
            x, y = (a[key].grad, b[key].grad) if gradients else (a[key], b[key])
            assert (x is None) == (y is None), key
            if x is not None:
                torch.testing.assert_close(x, y, atol=0, rtol=0, msg=lambda msg: key + ': ' + msg)
                count += 1
        return count

    results = {'scope': 'CPU float32，两层模型，实际小VAE和原始training_loss，多步AdamW；无真实资产、CUDA、DeepSpeed或数据解码',
               'torch': torch.__version__, 'cases': []}
    cases = [(v, b, .5) for v in ['uncond', 'joint', 'idm'] for b in [1, 2]]
    cases += [('idm', 2, 0.), ('idm', 2, 1.)]
    for variant, batch, cond_prob in cases:
        nets, rngs, optimizers = {}, {}, {}
        for name in roots:
            torch.manual_seed(42)
            nets[name] = model(name, variant, cond_prob)
            rngs[name] = torch.get_rng_state()
            params = [p for p in nets[name].parameters() if p.requires_grad]
            optimizers[name] = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-2, betas=(.9, .95))
        parameter_count = compare_named(nets['candidate'], nets['reference'])
        assert torch.equal(rngs['candidate'], rngs['reference'])
        gas = 2 if variant == 'uncond' else 1
        losses = []
        for update in range(args.steps):
            for micro in range(gas):
                generator = torch.Generator().manual_seed(700 + update * gas + micro)
                sample = {
                    'video': torch.randn(batch, 3, 9, 32, 64, generator=generator),
                    'action': torch.randn(batch, 32, 7, generator=generator),
                    'proprio': torch.randn(batch, 32, 8, generator=generator),
                    'context': torch.randn(batch, 5, 20, generator=generator),
                    'context_mask': torch.tensor([[True, True, True, False, False]]).expand(batch, -1),
                    'image_is_pad': torch.tensor([[False] * 6 + [True] * 3]).expand(batch, -1),
                    'action_is_pad': torch.tensor([[False] * 23 + [True] * 9]).expand(batch, -1),
                }
                outputs = {}
                for name in roots:
                    torch.set_rng_state(rngs[name])
                    loss, metrics = nets[name].training_loss(sample)
                    (loss / gas).backward()
                    rngs[name] = torch.get_rng_state()
                    outputs[name] = (float(loss.detach()), metrics)
                assert outputs['candidate'] == outputs['reference'], (variant, update, micro, outputs)
                assert torch.equal(rngs['candidate'], rngs['reference']), '随机流发生分歧'
                gradient_count = compare_named(nets['candidate'], nets['reference'], gradients=True)
                losses.append({'update': update + 1, 'microbatch': micro + 1, 'loss': outputs['candidate'][0],
                               'metrics': outputs['candidate'][1], 'loss_max_abs_diff': 0.0,
                               'gradient_tensors': gradient_count, 'gradient_max_abs_diff': 0.0})
            for name in roots:
                optimizers[name].step()
                optimizers[name].zero_grad(set_to_none=True)
            compare_named(nets['candidate'], nets['reference'])
        assert not hasattr(nets['candidate'], '_vae_encode_compiled')
        results['cases'].append({'variant': variant, 'batch': batch, 'gas': gas,
                                 'idm_cond_noise_prob': cond_prob if variant == 'idm' else None,
                                 'initial_parameter_tensors': parameter_count, 'updates': args.steps,
                                 'parameter_max_abs_diff_after_each_update': 0.0, 'losses': losses})
    payload = json.dumps(results, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()

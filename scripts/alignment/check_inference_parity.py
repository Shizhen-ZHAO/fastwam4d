"""三种模型的实际 eager 推理对照；无真实权重、T5 或模拟器。"""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
import torch.utils.checkpoint
from check_model_parity import package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-repo', required=True, type=Path)
    parser.add_argument('--target-repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
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

    results = {'scope': 'CPU float32, actual two-layer DiT/MoT/small VAE, supplied context; no pretrained T5, CUDA, compile or LIBERO', 'cases': []}
    for variant in ['uncond', 'joint', 'idm']:
        nets = {}
        for name in roots:
            torch.manual_seed(42)
            nets[name] = model(name, variant, .5).eval()
        for seed, steps, shift in [(42, 10, None), (42, 4, None), (7, 1, 3.), (42, 20, 5.)]:
            g = torch.Generator().manual_seed(991)
            kwargs = dict(prompt=None, input_image=torch.randn(1, 3, 32, 64, generator=g),
                          action_horizon=32, proprio=torch.randn(1, 8, generator=g),
                          context=torch.randn(1, 5, 20, generator=g),
                          context_mask=torch.tensor([[True, True, True, False, False]]),
                          num_inference_steps=steps, seed=seed, sigma_shift=shift)
            if steps == 10:
                # 正式评测 encode_prompt 将 padding 的 context 置零，再返回全 True mask。
                kwargs['context'][:, 3:] = 0
                kwargs['context_mask'] = torch.ones(1, 5, dtype=torch.bool)
            for method in ['infer_action', 'infer_joint']:
                outputs, traces, rngs = {}, {}, {}
                for name, net in nets.items():
                    trace = []
                    originals = []
                    # 记录两条 scheduler 的每次速度预测、输入和更新结果。
                    for kind in ['video', 'action']:
                        scheduler = getattr(net, 'infer_' + kind + '_scheduler')
                        original = scheduler.step
                        def step_fn(pred, delta, latent, original=original, kind=kind):
                            value = original(pred, delta, latent)
                            trace.append((kind, pred.detach().clone(), delta.detach().clone(),
                                          latent.detach().clone(), value.detach().clone()))
                            return value
                        scheduler.step = step_fn
                        originals.append((scheduler, original))
                    call = dict(kwargs)
                    signature = inspect.signature(getattr(net, method))
                    if 'num_video_frames' in signature.parameters:
                        call['num_video_frames'] = 9
                    if 'compile_action_infer' in signature.parameters:
                        call['compile_action_infer'] = False
                    if method == 'infer_joint':
                        call['test_action_with_infer_action'] = False
                    torch.manual_seed(1234)
                    with torch.no_grad():
                        outputs[name] = getattr(net, method)(**call)
                    rngs[name] = torch.get_rng_state()
                    traces[name] = trace
                    for scheduler, original in originals:
                        scheduler.step = original
                left, right = outputs['candidate'], outputs['reference']
                torch.testing.assert_close(left['action'], right['action'], atol=0, rtol=0)
                assert torch.equal(rngs['candidate'], rngs['reference'])
                a, b = traces['candidate'], traces['reference']
                assert len(a) == len(b), (variant, method, len(a), len(b))
                for index, (x, y) in enumerate(zip(a, b)):
                    assert x[0] == y[0]
                    for u, v in zip(x[1:], y[1:]):
                        torch.testing.assert_close(u, v, atol=0, rtol=0, msg=lambda msg: f'{variant}/{method}/{index}: {msg}')
                if method == 'infer_joint':
                    assert len(left['video']) == len(right['video']) == 9
                    for x, y in zip(left['video'], right['video']):
                        assert np.array_equal(np.asarray(x), np.asarray(y))
                results['cases'].append(dict(variant=variant, method=method, seed=seed,
                                             steps=steps, shift=shift, scheduler_updates=len(a),
                                             action_max_abs_diff=0., all_scheduler_tensors_equal=True,
                                             decoded_video_equal=method == 'infer_joint'))
    text = json.dumps(results, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Load PI0.5 once and evaluate a sequence of Sorting servers."""
from __future__ import annotations
import argparse, socket, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'lerobot/src'),str(ROOT/'scripts')]
from lerobot.configs import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from pi05_sorting_client import image_tensor, policy_state, seed_policy
from wire_protocol import recv_obj, send_obj
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--dataset-root',required=True); p.add_argument('--repo-id',required=True); p.add_argument('--ports',type=int,nargs='+',required=True); p.add_argument('--policy-seeds',type=int,nargs='+',required=True); p.add_argument('--n-action-steps',type=int,required=True); a=p.parse_args()
 if len(a.ports)!=len(a.policy_seeds): raise ValueError('ports and seeds differ')
 seed_policy(a.policy_seeds[0]); ds=LeRobotDataset(a.repo_id,root=a.dataset_root,download_videos=False)
 cfg=PI05Config(pretrained_path=a.model,device='cuda',dtype='bfloat16',n_action_steps=a.n_action_steps,empty_cameras=1,use_relative_actions=False,normalization_mapping={'VISUAL':NormalizationMode.IDENTITY,'STATE':NormalizationMode.QUANTILES,'ACTION':NormalizationMode.QUANTILES})
 policy=make_policy(cfg,ds_meta=ds.meta); pre,post=make_pre_post_processors(cfg,dataset_stats=ds.meta.stats,dataset_meta=ds.meta); policy.eval(); print(f'loaded once; n={a.n_action_steps}',flush=True)
 for port,seed in zip(a.ports,a.policy_seeds,strict=True):
  seed_policy(seed); policy.reset(); s=socket.create_connection(('127.0.0.1',port),timeout=30); s.settimeout(None)
  while True:
   m=recv_obj(s)
   if m['type']=='result': print('EPISODE_RESULT='+str(m),flush=True); break
   o=m['observation']; expected_dim=ds.meta.features['observation.state']['shape'][0]; raw={'observation.images.front':image_tensor(o['observation.images.front']),'observation.images.wrist':image_tensor(o['observation.images.wrist']),'observation.state':policy_state(o['observation.state'],expected_dim),'task':m['task']}
   with torch.inference_mode(): x=post(policy.select_action(pre(raw)))
   send_obj(s,{'type':'action','action':np.asarray(x.cpu(),np.float32).reshape(-1)})
  s.close()
if __name__=='__main__': main()

from ..T2M.options import option_transformer as option_trans
import clip
import torch
from ..T2M.models import vqvae as vqvae
from ..T2M.models import t2m_trans as trans

from T2M.utils.motion_process import recover_from_ric

import warnings
import numpy as np


def infer_motion(text="a person is jumping",
                 resume_pth='pretrained/VQVAE/net_last.pth',
                 resume_trans='pretrained/VQTransformer_corruption05/net_best_fid.pth',
                 mean='path to /mean.npy',
                 std='path to /std.npy',
                 ):

    clip_text = [text]
    args = option_trans.get_args_parser()

    args.dataname = 't2m'
    args.resume_pth = resume_pth
    args.resume_trans = resume_trans
    args.down_t = 2
    args.depth = 3
    args.block_size = 51

    warnings.filterwarnings('ignore')

    # load clip model and datasets
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'),
                                            jit=True)  # Must set jit=False for training
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    net = vqvae.HumanVQVAE(args,  # use args to define different parameters in different quantizers
                           args.nb_code,
                           args.code_dim,
                           args.output_emb_width,
                           args.down_t,
                           args.stride_t,
                           args.width,
                           args.depth,
                           args.dilation_growth_rate)

    trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                                  embed_dim=1024,
                                                  clip_dim=args.clip_dim,
                                                  block_size=args.block_size,
                                                  num_layers=9,
                                                  n_head=16,
                                                  drop_out_rate=args.drop_out_rate,
                                                  fc_rate=args.ff_rate)

    print('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval()
    trans_encoder.cuda()

    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    pred_pose = net.forward_decoder(index_motion)

    mean = torch.from_numpy(np.load(mean)).cuda()
    std = torch.from_numpy(np.load(std)).cuda()

    pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)
    xyz = xyz[0]
    return xyz


def retrieve_marker_locations(joints):
    data = joints.copy().reshape(len(joints), -1, 3)

    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7],
                          [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                                                                  [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                                                                  [9, 13, 16, 18, 20]]
    mins = data.min(axis=0).min(axis=0)

    parts = ["RightLeg", "LeftLeg", "Spine", "RightArm", "LeftArm"]

    height_offset = mins[1]
    data[:, :, 1] -= height_offset

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    frames = []

    for index in range(data.shape[0]):
        for i, chain in enumerate(smpl_kinetic_chain):
            for idx, point in enumerate(data[index, chain]):
                frames.append({"x": point[0],
                               "y": point[1],
                               "z": point[2],
                               "Named": f"{parts[i]}-{idx}",
                               "ID": [i, idx]})
                print(f"[POINT {parts[i]}-{idx}] - loc({point[0]}, {point[1]}, {point[2]})")

    return {"frames": frames}


# Example Usage
if __name__ == "__main__":
    xyz = infer_motion()
    frames = retrieve_marker_locations(xyz)

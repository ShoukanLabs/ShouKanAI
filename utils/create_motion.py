from ..T2M.options import option_transformer as option_trans
import clip
import torch
from ..T2M.models import vqvae as vqvae
from ..T2M.models import t2m_trans as trans

from T2M.utils.motion_process import recover_from_ric

import warnings
import numpy as np


class MotionManager:
    def __int__(self,
                resume_pth='pretrained/VQVAE/net_last.pth',
                resume_trans='pretrained/VQTransformer_corruption05/net_best_fid.pth',
                mean='path to /mean.npy',
                std='path to /std.npy'):

        self.args = option_trans.get_args_parser()

        self.args.dataname = 't2m'
        self.args.resume_pth = resume_pth
        self.args.resume_trans = resume_trans
        self.args.down_t = 2
        self.args.depth = 3
        self.args.block_size = 51

        warnings.filterwarnings('ignore')

        self.net = vqvae.HumanVQVAE(self.args,  # use args to define different parameters in different quantizers
                                    self.args.nb_code,
                                    self.args.code_dim,
                                    self.args.output_emb_width,
                                    self.args.down_t,
                                    self.args.stride_t,
                                    self.args.width,
                                    self.args.depth,
                                    self.args.dilation_growth_rate)

        self.trans_encoder = trans.Text2Motion_Transformer(num_vq=self.args.nb_code,
                                                           embed_dim=1024,
                                                           clip_dim=self.args.clip_dim,
                                                           block_size=self.args.block_size,
                                                           num_layers=9,
                                                           n_head=16,
                                                           drop_out_rate=self.args.drop_out_rate,
                                                           fc_rate=self.args.ff_rate)

        print('[LOADING] loading checkpoint from {}'.format(self.args.resume_pth))
        self.ckpt = torch.load(self.args.resume_pth, map_location='cpu')
        self.net.load_state_dict(self.ckpt['net'], strict=True)
        self.net.eval()
        self.net.cuda()

        print('[LOADING] loading transformer checkpoint from {}'.format(self.args.resume_trans))
        self.ckpt = torch.load(self.args.resume_trans, map_location='cpu')
        self.trans_encoder.load_state_dict(self.ckpt['trans'], strict=True)
        self.trans_encoder.eval()
        self.trans_encoder.cuda()

        self.mean = torch.from_numpy(np.load(mean)).cuda()
        self.std = torch.from_numpy(np.load(std)).cuda()

        print("[READY] Motion manager has been initialised")

    def infer_motion(self, text="a person is jumping"):
        clip_text = [text]

        # load clip model and datasets
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'),
                                                jit=True)  # Must set jit=False for training
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        text = clip.tokenize(clip_text, truncate=True).cuda()
        feat_clip_text = clip_model.encode_text(text).float()
        index_motion = self.trans_encoder.sample(feat_clip_text[0:1], False)
        pred_pose = self.net.forward_decoder(index_motion)

        pred_xyz = recover_from_ric((pred_pose * self.std + self.mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)
        xyz = xyz[0]

        del clip_preprocess, clip_model, self.net, self.ckpt, self.trans_encoder, self.mean, self.std

        torch.cuda.empty_cache()

        print("[READY] Motion has been synthesized and is ready for conversion to trackers")

        return xyz

    @staticmethod
    def retrieve_marker_locations(joints):
        print(f"[STARTING] ")
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

        print("[READY] Motion returned as locations")
        return {"frames": frames}

    def retrieve_marker_locations_from_saved(self, path):
        print(f"[LOADING] Loading {path}")
        joints = np.load(path)
        print(f"[STARTING] Beginning conversion of {path}")
        return self.retrieve_marker_locations(joints)


# Example Usage
if __name__ == "__main__":
    manager = MotionManager("path to pth", "path to trans", "path to mean", "path to std")
    # xyzloc = manager.infer_motion("a person is jumping")
    # finalframes = manager.retrieve_marker_locations(xyzloc)

    frames = manager.retrieve_marker_locations_from_saved(r"D:\Github\ShouKanAI\Save For Later\motion.npy")

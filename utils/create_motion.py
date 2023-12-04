import json
from T2M.options import option_transformer as option_trans
import clip
import torch
from T2M.models import vqvae as vqvae
from T2M.models import t2m_trans as trans

from T2M.utils.motion_process import recover_from_ric

import warnings
import numpy as np

from scipy.spatial.transform import Rotation as R


class MotionManager:
    def __init__(self,
                 ignore_load=False,
                 resume_pth='pretrained/VQVAE/net_last.pth',
                 resume_trans='pretrained/VQTransformer_corruption05/net_best_fid.pth',
                 mean='path to /mean.npy',
                 std='path to /std.npy'):

        self.ignore_load = ignore_load

        if not ignore_load:
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
        if not self.ignore_load:
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
        else:
            print("[WARNING] Manager running without loaded models, cannot infer motion")
            return

    @staticmethod
    def retrieve_markers(joints):
        print(f"[STARTING] ")
        data = joints.copy().reshape(len(joints), -1, 3)

        framesTotal = data.shape[0]

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
        markers = []

        for index in range(framesTotal):
            frames.append([])
            for i, chain in enumerate(smpl_kinetic_chain):
                for idx, point in enumerate(data[index, chain]):
                    frameDict = {f"{i}{idx}": {"x": point[0],
                                               "y": point[1],
                                               "z": point[2],
                                               "Named": f"{parts[i]}-{idx}",
                                               }}
                    frames[index].append(frameDict)
                    print(f"[{index} - POINT {parts[i]}-{idx}] - loc({point[0]}, {point[1]}, {point[2]})")

        for frame in frames:
            def _get_np_loc(name, listType=False, xyz=False):
                for i in frame:
                    if list(i.keys())[0] == name:
                        if xyz:
                            return {"x": i[name]["x"],
                                    "y": i[name]["y"],
                                    "z": i[name]["z"]}
                        elif listType:
                            return [i[name]["x"],
                                    i[name]["y"],
                                    i[name]["z"]]
                        else:
                            return np.array([i[name]["x"],
                                             i[name]["y"],
                                             i[name]["z"]])

            def _only_loc(name):
                return {"location": _get_np_loc(name, xyz=True),
                        "rotation": {"x": 0, "y": 0, "z": 0}}

            hip_left = "11"
            hip_right = "01"
            hip_up = "20"

            knee_left = "12"
            knee_right = "02"

            ankle_left = "13"
            ankle_right = "03"

            # hip
            x = _get_np_loc(hip_right) - _get_np_loc(hip_left)
            w = _get_np_loc(hip_up) - _get_np_loc(hip_left)
            z = np.cross(x, w)
            y = np.cross(z, x)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            hip_rot = np.vstack((x, y, z)).T

            # right leg

            y = _get_np_loc(knee_right) - _get_np_loc(ankle_right)
            w = _get_np_loc(hip_right) - _get_np_loc(ankle_right)
            z = np.cross(w, y)
            if np.sqrt(sum(z ** 2)) < 1e-6:
                w = _get_np_loc(hip_left) - _get_np_loc(ankle_left)
                z = np.cross(w, y)
            x = np.cross(y, z)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            leg_r_rot = np.vstack((x, y, z)).T

            # left leg

            y = _get_np_loc(knee_left) - _get_np_loc(ankle_left)
            w = _get_np_loc(hip_left) - _get_np_loc(ankle_left)
            z = np.cross(w, y)
            if np.sqrt(sum(z ** 2)) < 1e-6:
                w = _get_np_loc(hip_right) - _get_np_loc(ankle_left)
                z = np.cross(w, y)
            x = np.cross(y, z)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            leg_l_rot = np.vstack((x, y, z)).T

            rot_hip = R.from_matrix(hip_rot).as_euler("xyz", degrees=True)
            rot_leg_r = R.from_matrix(leg_r_rot).as_euler("xyz", degrees=True)
            rot_leg_l = R.from_matrix(leg_l_rot).as_euler("xyz", degrees=True)

            h = _get_np_loc(hip_up, listType=True, xyz=True)
            kl = _get_np_loc(knee_left, listType=True, xyz=True)
            kr = _get_np_loc(knee_right, listType=True, xyz=True)

            rxh, ryh, rzh = list(rot_hip)
            rxkl, rykl, rzkl = list(rot_leg_l)
            rxkr, rykr, rzkr = list(rot_leg_r)

            marker_dict = {
                "hip": {"location": h,
                        "rotation": {"x": rxh, "y": ryh, "z": rzh}},
                "leg_l": {"location": kl,
                          "rotation": {"x": rxkl, "y": rykl, "z": rzkl}},
                "leg_r": {"location": kr,
                          "rotation": {"x": rxkr, "y": rykr, "z": rzkr}},
                "foot_l": _only_loc("14"),
                "foot_r": _only_loc("04")
            }

            print(f"[{index} - MARKERS ADDED] \n{marker_dict}\n\n")
            markers.append(marker_dict)

        print("[READY] Motion returned as locations and emulated trackers")
        return {"frames": frames, "markers": markers}

    def retrieve_markers_from_saved(self, path):
        print(f"[LOADING] Loading {path}")
        joints = np.load(path)
        joints = joints[0]
        print(f"[STARTING] Beginning conversion of {path}")
        return self.retrieve_markers(joints)


# Example Usage
if __name__ == "__main__":

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)


    # manager = MotionManager("path to pth", "path to trans", "path to mean", "path to std")
    # xyzloc = manager.infer_motion("a person is jumping")
    # finalframes = manager.retrieve_marker_locations(xyzloc)

    manager = MotionManager(ignore_load=True)
    frames = manager.retrieve_markers_from_saved(r"D:\Github\ShouKanAI\Save For Later\motion.npy")

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(frames, f, indent=4, cls=NpEncoder)

import json
from T2M.options import option_transformer as option_trans
import clip
import torch
from T2M.models import vqvae as vqvae
from T2M.models import t2m_trans as trans

from ursina import *
from ursina.shaders import *
from T2M.utils.motion_process import recover_from_ric

import warnings
import numpy as np

from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

from pythonosc import udp_client

from steamvr_manager import SteamVRDeviceManager, force_tracker_cam

client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# create a window
window.borderless = False
window.title = 'Motion Debug'
visualizer = Ursina(vsync=True, use_ingame_console=False)

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
        maxs = data.max(axis=0).max(axis=0)

        parts = ["RightLeg", "LeftLeg", "Spine", "RightArm", "LeftArm"]

        height_offset = mins[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]

        frames = []
        markers = []

        for index in tqdm(range(framesTotal)):

            minx = mins[0] - trajec[index, 0]
            maxx = maxs[0] - trajec[index, 0]
            minz = mins[2] - trajec[index, 1]
            maxz = maxs[2] - trajec[index, 1]

            x = (minx + minx + maxx + maxx) / 4
            y = (0 + 0 + 0 + 0) / 4
            z = (minz + maxz + maxz + minz) / 4

            frames.append([])
            for i, chain in enumerate(smpl_kinetic_chain):
                for idx, point in enumerate(data[index, chain]):
                    frameDict = {f"{i}{idx}": {"x": point[0],
                                               "y": point[1],
                                               "z": point[2],
                                               "Named": f"{parts[i]}-{idx}",
                                               }}
                    frames[index].append(frameDict)

            global_offset = {f"global": {"x": x,
                                         "y": y,
                                         "z": z,
                                         "Named": "Global Offset",
                                         }}
            frames[index].append(global_offset)

        for idx, frame in tqdm(enumerate(frames)):
            markers.append([])

            def _get_np_loc(name, listType=False, xyz=False):
                for i in frame:
                    if list(i.keys())[0] == name:
                        if xyz:
                            return {"x": i[name]["x"] - frame[-1]["global"]["x"],
                                    "y": i[name]["y"] - frame[-1]["global"]["y"],
                                    "z": i[name]["z"] - frame[-1]["global"]["z"]}
                        elif listType:
                            return [i[name]["x"] - frame[-1]["global"]["x"],
                                    i[name]["y"] - frame[-1]["global"]["y"],
                                    i[name]["z"] - frame[-1]["global"]["z"]]
                        else:
                            return np.array([i[name]["x"] - frame[-1]["global"]["x"],
                                             i[name]["y"] - frame[-1]["global"]["y"],
                                             i[name]["z"] - frame[-1]["global"]["z"]])

            def _only_loc(name, rotation=None, qrot=None):
                if rotation is None:
                    return {"location": _get_np_loc(name, xyz=True),
                            "rotation": {"x": 0, "y": 0, "z": 0}}
                else:
                    rx, ry, rz = list(rotation)

                    return {"location": _get_np_loc(name, xyz=True),
                            "rotation": {"x": rx, "y": ry, "z": rz}, "qrotation": qrot}

            hip_left = "11"
            hip_right = "01"
            hip_up = "20"

            chest_left = "41"
            chest_right = "31"
            chest_down = "22"

            knee_left = "12"
            knee_right = "02"

            ankle_left = "13"
            ankle_right = "03"

            foot_right = "04"
            foot_left = "14"

            elbow_right = "33"
            elbow_left = "43"

            hand_right = "34"
            hand_left = "44"

            head_top = "25"
            neck = "24"

            # head
            y = _get_np_loc(head_top) - _get_np_loc(neck)
            w = _get_np_loc(neck) - _get_np_loc(head_top)
            z = np.cross(w, y)
            if np.sqrt(sum(z ** 2)) < 1e-6:
                w = _get_np_loc(chest_left) - _get_np_loc(neck)
                z = np.cross(w, y)
            x = np.cross(y, z)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            head_rot = np.vstack((x, y, z)).T

            # hip
            x = _get_np_loc(hip_right) - _get_np_loc(hip_left)
            w = _get_np_loc(hip_up) - _get_np_loc(hip_left)
            z = np.cross(x, w)
            y = np.cross(z, x)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            hip_rot = np.vstack((x, y, z)).T

            # chest
            x = _get_np_loc(chest_right) - _get_np_loc(chest_left)
            w = _get_np_loc(chest_down) - _get_np_loc(chest_left)
            z = np.cross(x, w)
            y = np.cross(z, x)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            chest_rot = np.vstack((x, y, z)).T

            # right leg
            y = _get_np_loc(knee_right) - _get_np_loc(ankle_right)
            w = _get_np_loc(ankle_right) - _get_np_loc(foot_right)
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
            w = _get_np_loc(ankle_left) - _get_np_loc(foot_left)
            z = np.cross(w, y)
            if np.sqrt(sum(z ** 2)) < 1e-6:
                w = _get_np_loc(hip_right) - _get_np_loc(ankle_right)
                z = np.cross(w, y)
            x = np.cross(y, z)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            leg_l_rot = np.vstack((x, y, z)).T

            # right arm
            y = _get_np_loc(elbow_right) - _get_np_loc(hand_right)
            w = _get_np_loc(elbow_right) - _get_np_loc(chest_right)
            z = np.cross(w, y)
            if np.sqrt(sum(z ** 2)) < 1e-6:
                w = _get_np_loc(chest_left) - _get_np_loc(hand_left)
                z = np.cross(w, y)
            x = np.cross(y, z)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            arm_r_rot = np.vstack((x, y, z)).T

            # left arm
            y = _get_np_loc(elbow_left) - _get_np_loc(hand_left)
            w = _get_np_loc(elbow_left) - _get_np_loc(chest_left)
            z = np.cross(w, y)
            if np.sqrt(sum(z ** 2)) < 1e-6:
                w = _get_np_loc(chest_left) - _get_np_loc(chest_right)
                z = np.cross(w, y)
            x = np.cross(y, z)

            x = x / np.sqrt(sum(x ** 2))
            y = y / np.sqrt(sum(y ** 2))
            z = z / np.sqrt(sum(z ** 2))

            arm_l_rot = np.vstack((x, y, z)).T

            # Correct and store
            rot_chest = R.from_matrix(chest_rot).as_euler("xyz", degrees=True)
            rot_hip = R.from_matrix(hip_rot).as_euler("xyz", degrees=True)
            rot_head = R.from_matrix(arm_l_rot).as_euler("xyz", degrees=True)
            rot_leg_r = R.from_matrix(leg_r_rot).as_euler("xyz", degrees=True)
            rot_leg_l = R.from_matrix(leg_l_rot).as_euler("xyz", degrees=True)
            rot_arm_r = R.from_matrix(arm_r_rot).as_euler("xyz", degrees=True)
            rot_arm_l = R.from_matrix(arm_l_rot).as_euler("xyz", degrees=True)

            qrot_chest = R.from_matrix(chest_rot).as_quat(True)
            qrot_hip = R.from_matrix(hip_rot).as_quat(True)
            qrot_head = R.from_matrix(arm_l_rot).as_quat(True)
            qrot_leg_r = R.from_matrix(leg_r_rot).as_quat(True)
            qrot_leg_l = R.from_matrix(leg_l_rot).as_quat(True)
            qrot_arm_r = R.from_matrix(arm_r_rot).as_quat(True)
            qrot_arm_l = R.from_matrix(arm_l_rot).as_quat(True)

            cd = _get_np_loc(chest_down, listType=True)
            cl = _get_np_loc(chest_left, listType=True)
            c = {"x": cd[0], "y": cl[1], "z": cd[2]}
            h = _get_np_loc(hip_up, listType=True, xyz=True)
            kl = _get_np_loc(knee_left, listType=True, xyz=True)
            kr = _get_np_loc(knee_right, listType=True, xyz=True)
            al = _get_np_loc(hand_left, listType=True, xyz=True)
            ar = _get_np_loc(hand_right, listType=True, xyz=True)
            hh = _get_np_loc(head_top, listType=True, xyz=True)

            rxc, ryc, rzc = list(rot_chest)
            rxh, ryh, rzh = list(rot_hip)
            _, _, rzhh = list(rot_head)
            rxkl, rykl, rzkl = list(rot_leg_l)
            rxkr, rykr, rzkr = list(rot_leg_r)
            rxal, ryal, rzal = list(rot_leg_l)
            rxar, ryar, rzar = list(rot_leg_r)

            marker_dict = {
                "head": {"location": hh,
                         "rotation": {"x": -rxc, "y": ryc, "z": rzhh + 30}, "qrotation": qrot_head},
                "chest": {"location": c,
                          "rotation": {"x": -rxc - 90, "y": ryc, "z": rzc}, "qrotation": qrot_chest},
                "hip": {"location": h,
                        "rotation": {"x": rxh + 90, "y": ryh, "z": rzh}, "qrotation": qrot_hip},
                "leg_r": {"location": kl,
                          "rotation": {"x": rxkl - 90, "y": rykl - 90, "z": rzkl}, "qrotation": qrot_leg_r},
                "leg_l": {"location": kr,
                          "rotation": {"x": rxkr - 90, "y": rykr - 90, "z": rzkr}, "qrotation": qrot_leg_l},
                "foot_r": _only_loc(foot_left, rot_leg_r, qrot_leg_r),
                "foot_l": _only_loc(foot_right, rot_leg_l, qrot_leg_l),
                "arm_r": {"location": al,
                          "rotation": {"x": rxal, "y": ryal - 60, "z": rzal}, "qrotation": qrot_arm_r},
                "arm_l": {"location": ar,
                          "rotation": {"x": rxar, "y": ryar + 110, "z": rzar}, "qrotation": qrot_arm_l},
            }
            markers[idx].append(marker_dict)

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

    # Visualiser
    # HTC Vive Tracker by Marco Romero
    # [CC-BY] (https://creativecommons.org/licenses/by/3.0/)
    # via Poly Pizza (https://poly.pizza/m/3g9sc265XVC)

    # HTC Vive Controller by serkanmert
    # [CC-BY] (https://creativecommons.org/licenses/by/4.0/)
    # via Sketchfab (https://sketchfab.com/3d-models/htc-vive-controller-f9cc5f021c044a25b2c89029448d9602)

    # HTC Vive Headset by Eternal Realm
    # [CC-BY] (https://creativecommons.org/licenses/by/4.0/)
    # via Sketchfab (https://sketchfab.com/3d-models/htc-vive-4818cdb261714a70a08991a3d4ed3749)

    scale = 0.001
    scale_control = 1
    scale_head = 0.03

    tracker_model = "./resources/HTC_Vive_Tracker.obj"
    controller_model = "./resources/HTC_Vive_Controller.obj"
    headset_model = "./resources/HTC_Vive_Headset.obj"

    head = Entity(model=headset_model,
                  color=color.dark_gray,
                  scale=(scale_head, scale_head, scale_head),
                  shader=basic_lighting_shader)

    chest = Entity(model=tracker_model,
                   color=color.dark_gray,
                   scale=(scale, scale, scale),
                   shader=basic_lighting_shader)

    hip = Entity(model=tracker_model,
                 color=color.dark_gray,
                 scale=(scale, scale, scale),
                 shader=basic_lighting_shader)

    legl = Entity(model=tracker_model,
                  color=color.dark_gray,
                  scale=(scale, scale, scale),
                  shader=basic_lighting_shader)

    legr = Entity(model=tracker_model,
                  color=color.dark_gray,
                  scale=(scale, scale, scale),
                  shader=basic_lighting_shader)

    footl = Entity(model=tracker_model,
                   color=color.dark_gray,
                   scale=(scale, scale, scale),
                   shader=basic_lighting_shader)

    footr = Entity(model=tracker_model,
                   color=color.dark_gray,
                   scale=(scale, scale, scale),
                   shader=basic_lighting_shader)

    arml = Entity(model=controller_model,
                  color=color.dark_gray,
                  scale=(scale_control, scale_control, scale_control),
                  shader=basic_lighting_shader)

    armr = Entity(model=controller_model,
                  color=color.dark_gray,
                  scale=(scale_control, scale_control, scale_control),
                  shader=basic_lighting_shader)

    chest.double_sided = True
    hip.double_sided = True
    legl.double_sided = True
    legr.double_sided = True
    footl.double_sided = True
    footr.double_sided = True
    arml.double_sided = True
    armr.double_sided = True
    head.double_sided = True

    frameidx = 0

    text_entity = Text("Frame: 0", world_scale=30, origin=(4.5, -11))

    force_tracker_cam(True)

    device_manager = SteamVRDeviceManager()

    time.sleep(10)

    device_manager.create_trackers()

    # test_frame = {
    #             "head": {
    #                 "location": {
    #                     "x": 0.3154,
    #                     "y": 0.4922,
    #                     "z": 0.7933
    #                 },
    #                 "rotation": {
    #                     "x": 109.45,
    #                     "y": -3.098,
    #                     "z": -8.253
    #                 }
    #             },
    #             "chest": {
    #                 "location": {
    #                     "x": 0.21068,
    #                     "y": 0.5717,
    #                     "z": 0.5125
    #                 },
    #                 "rotation": {
    #                     "x": 19.459,
    #                     "y": -3.098,
    #                     "z": -172.7
    #                 }
    #             },
    #             "hip": {
    #                 "location": {
    #                     "x": 0.13341,
    #                     "y": 0.7318,
    #                     "z": 0.3222
    #                 },
    #                 "rotation": {
    #                     "x": 138.4,
    #                     "y": -15.718,
    #                     "z": -162.94
    #                 }
    #             },
    #             "leg_r": {
    #                 "location": {
    #                     "x": 0.25947,
    #                     "y": 1.0151,
    #                     "z": 0.018246
    #                 },
    #                 "rotation": {
    #                     "x": -84.18,
    #                     "y": -143.8,
    #                     "z": -175.13
    #                 }
    #             },
    #             "leg_l": {
    #                 "location": {
    #                     "x": -0.02539,
    #                     "y": 0.9341,
    #                     "z": 0.17416
    #                 },
    #                 "rotation": {
    #                     "x": -241.0,
    #                     "y": -145.7,
    #                     "z": -30.50
    #                 }
    #             },
    #             "foot_r": {
    #                 "location": {
    #                     "x": 0.21378,
    #                     "y": 1.6113,
    #                     "z": -0.0919
    #                 },
    #                 "rotation": {
    #                     "x": 5.814,
    #                     "y": -53.87,
    #                     "z": -175.13
    #                 }
    #             },
    #             "foot_l": {
    #                 "location": {
    #                     "x": -0.0015141,
    #                     "y": 1.4202,
    #                     "z": 0.2657
    #                 },
    #                 "rotation": {
    #                     "x": -151.0,
    #                     "y": -55.718,
    #                     "z": -30.50
    #                 }
    #             },
    #             "arm_r": {
    #                 "location": {
    #                     "x": 0.34248,
    #                     "y": 0.28329,
    #                     "z": 0.39254
    #                 },
    #                 "rotation": {
    #                     "x": 5.814,
    #                     "y": -113.87,
    #                     "z": -175.13
    #                 }
    #             },
    #             "arm_l": {
    #                 "location": {
    #                     "x": 0.09902,
    #                     "y": 0.27765,
    #                     "z": 0.4460
    #                 },
    #                 "rotation": {
    #                     "x": -151.0,
    #                     "y": 54.281,
    #                     "z": -30.50
    #                 }
    #             }
    #         }
    #
    # device_manager.update_pose_full(test_frame)

    def update():
        global frameidx
        frameidx += 1 * time.dt * 25
        if frameidx > len(frames["markers"]) - 1:
            frameidx = 0

        frameidxdelta = int(frameidx)

        text_entity.text = f"Frame: {frameidxdelta}"

        frame = frames["markers"][frameidxdelta]
        frame = frame[0]
        device_manager.update_pose_full(frame, velocity=frameidx)

        # Positions
        head.position = Vec3(frame["head"]["location"]["x"],
                             frame["head"]["location"]["y"],
                             frame["head"]["location"]["z"])

        chest.position = Vec3(frame["chest"]["location"]["x"],
                              frame["chest"]["location"]["y"],
                              frame["chest"]["location"]["z"])

        hip.position = Vec3(frame["hip"]["location"]["x"],
                            frame["hip"]["location"]["y"],
                            frame["hip"]["location"]["z"])

        legl.position = Vec3(frame["leg_l"]["location"]["x"],
                             frame["leg_l"]["location"]["y"],
                             frame["leg_l"]["location"]["z"])

        legr.position = Vec3(frame["leg_r"]["location"]["x"],
                             frame["leg_r"]["location"]["y"],
                             frame["leg_r"]["location"]["z"])

        footl.position = Vec3(frame["foot_l"]["location"]["x"],
                              frame["foot_l"]["location"]["y"],
                              frame["foot_l"]["location"]["z"])

        footr.position = Vec3(frame["foot_r"]["location"]["x"],
                              frame["foot_r"]["location"]["y"],
                              frame["foot_r"]["location"]["z"])

        arml.position = Vec3(frame["arm_l"]["location"]["x"],
                             frame["arm_l"]["location"]["y"],
                             frame["arm_l"]["location"]["z"])

        armr.position = Vec3(frame["arm_r"]["location"]["x"],
                             frame["arm_r"]["location"]["y"],
                             frame["arm_r"]["location"]["z"])

        # Rotations
        head.rotation = Vec3(frame["head"]["rotation"]["x"],
                             frame["head"]["rotation"]["y"],
                             frame["head"]["rotation"]["z"])

        chest.rotation = Vec3(frame["chest"]["rotation"]["x"],
                              frame["chest"]["rotation"]["y"],
                              frame["chest"]["rotation"]["z"])

        hip.rotation = Vec3(frame["hip"]["rotation"]["x"],
                            frame["hip"]["rotation"]["y"],
                            frame["hip"]["rotation"]["z"])

        legl.rotation = Vec3(-frame["leg_l"]["rotation"]["x"],
                             -frame["leg_l"]["rotation"]["y"],
                             -frame["leg_l"]["rotation"]["z"])

        legr.rotation = Vec3(-frame["leg_r"]["rotation"]["x"],
                             -frame["leg_r"]["rotation"]["y"],
                             -frame["leg_r"]["rotation"]["z"])

        footl.rotation = Vec3(frame["foot_l"]["rotation"]["x"],
                              frame["foot_l"]["rotation"]["y"],
                              frame["foot_l"]["rotation"]["z"])

        footr.rotation = Vec3(frame["foot_r"]["rotation"]["x"],
                              frame["foot_r"]["rotation"]["y"],
                              frame["foot_r"]["rotation"]["z"])

        arml.rotation = Vec3(frame["arm_l"]["rotation"]["x"],
                             frame["arm_l"]["rotation"]["y"],
                             frame["arm_l"]["rotation"]["z"])

        armr.rotation = Vec3(frame["arm_r"]["rotation"]["x"],
                             frame["arm_r"]["rotation"]["y"],
                             frame["arm_r"]["rotation"]["z"])


    ec = EditorCamera(ignore_scroll_on_ui=True)
    front = Entity(model="cube", scale=(3, 3, 0.01), shader=basic_lighting_shader, rotation_x=90)

    pivot = Entity()
    DirectionalLight(parent=pivot, y=5, z=3, shadows=True)

    visualizer.run()

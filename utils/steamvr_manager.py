import json
import subprocess
import struct
import socket
import time
import numpy as np
import pyrr
from scipy.spatial.transform import Rotation as R

# deprecated but I'll keep it around since it`s useful
def force_driver(set_driver_state=False,
                 settings_path=r"C:\Program Files (x86)\Steam\config\steamvr.vrsettings",
                 driver_settings_path=r"C:\Program Files (x86)\Steam\steamapps\common\SteamVR\drivers\null\resources\settings\default.vrsettings"):
    # force null driver
    with open(settings_path, "r") as jsonFile:
        data = json.load(jsonFile)
        with open(f"{settings_path}.backup", "w") as jsonFile:
            json.dump(data, jsonFile)

    data["steamvr"]["requireHmd"] = not set_driver_state
    data["steamvr"]["activateMultipleDrivers"] = set_driver_state
    if set_driver_state:
        data["steamvr"]["forcedDriver"] = "null"
    else:
        data["steamvr"]["forcedDriver"] = ""

    with open(settings_path, "w") as jsonFile:
        json.dump(data, jsonFile)

    # activate null driver
    with open(driver_settings_path, "r") as jsonFile:
        data = json.load(jsonFile)
        with open(f"{driver_settings_path}.backup", "w") as jsonFile:
            json.dump(data, jsonFile)

    data["driver_null"]["enable"] = set_driver_state

    with open(driver_settings_path, "w") as jsonFile:
        json.dump(data, jsonFile)


def force_tracker_cam(set_state=False,
                      settings_path=r"C:\Program Files (x86)\Steam\config\steamvr.vrsettings",
                      driver_settings_path=r"C:\Program Files (x86)\Steam\steamapps\common\SteamVR\drivers\null\resources\settings\default.vrsettings"):
    # force null driver
    with open(settings_path, "r") as jsonFile:
        data = json.load(jsonFile)
        with open(f"{settings_path}.backup", "w") as jsonFile:
            json.dump(data, jsonFile)

    if set_state:
        data["trackers"] = {
            "/devices/hobovr/t0": "TrackerRole_Camera",
            "/devices/hobovr/t1": "TrackerRole_Chest",
            "/devices/hobovr/t2": "TrackerRole_Waist",
            "/devices/hobovr/t3": "TrackerRole_LeftKnee",
            "/devices/hobovr/t4": "TrackerRole_RightKnee",
            "/devices/hobovr/t5": "TrackerRole_LeftFoot",
            "/devices/hobovr/t6": "TrackerRole_RightFoot"
        }
        data["TrackingOverrides"] = {"/devices/hobovr/t0": "/user/head"}
    else:
        data["trackers"] = {}
        data["TrackingOverrides"] = {}

    with open(f"{settings_path}", "w") as jsonFile:
        json.dump(data, jsonFile)

    force_driver(set_state, settings_path, driver_settings_path)

class SteamVRDeviceManager:
    def __init__(self):
        # API Structs
        self.TERMINATOR = b'\n'
        self.SEND_TERMINATOR = b'\t\r\n'
        self.MANAGER_UDU_MSG_t = struct.Struct("130I")
        self.POSE_t = struct.Struct("13f")
        self.CONTROLLER_t = struct.Struct("22f")

        # Networking
        self.manager_socket = None
        self.tracking_socket = None
        self.resp_a = None
        self.resp_b = None
        self.id_msg_b = None
        self.id_msg_a = None
        self.client_b = None
        self.client_a = None
        self.serversocket = None

        # Devices
        self.device_list = None

        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind(('', 6969))
        self.serversocket.listen(2)  # driver connects with 2 sockets

        self.launch_steamvr()

        print("[DRIVER] Waiting for driver to connect...")

        self.client_a = self.serversocket.accept()
        self.client_b = self.serversocket.accept()

        print("[DRIVER] Waiting for driver resolution...")

        self.resp_a = self.client_a[0].recv(50)
        self.resp_b = self.client_b[0].recv(50)

        if self.TERMINATOR in self.resp_a:
            self.id_msg_a, self.resp_a = self.resp_a.split(self.TERMINATOR, 1)

        if self.TERMINATOR in self.resp_b:
            self.id_msg_b, self.resp_b = self.resp_b.split(self.TERMINATOR, 1)

        if self.id_msg_a == b"hello" and self.id_msg_b == b"monky":  # idk why these are named this way
            self.tracking_socket = self.client_a[0]
            self.manager_socket = self.client_b[0]

        elif self.id_msg_b == b"hello" and self.id_msg_a == b"monky":  # idk why these are named this way
            self.tracking_socket = self.client_b[0]
            self.manager_socket = self.client_a[0]

        else:
            print("[DRIVER ERROR] Bad connection, try re-instantiating")
            self.client_a[0].close()
            self.client_b[0].close()

            self.serversocket.close()

    @staticmethod
    def launch_steamvr():
        subprocess.run("start steam://run/250820", shell=True)

    def create_trackers(self):
        self.device_list = self.MANAGER_UDU_MSG_t.pack(
            20,  # HobovrManagerMsgType::Emsg_uduString
            9,  # 6 devices - 1 hmd, 2 controllers, 6 trackers
            2, 13,  # device description - /devices/hobovr/t0
            1, 22,  # device description
            1, 22,  # device description
            2, 13,  # device description
            2, 13,  # device description
            2, 13,  # device description
            2, 13,  # device description
            2, 13,  # device description
            2, 13,  # device description
            *np.zeros((128 - 2 * 9), dtype=int)
        )

        self.manager_socket.sendall(self.device_list + self.SEND_TERMINATOR)

    def update_pose_full(self, frame, app_menu=False, click_trigger=False, velocity=1, scale=1, rot_scale=1):
        packet = b''

        lx = frame["head"]["location"]["x"] * scale
        ly = frame["head"]["location"]["y"] * scale
        lz = frame["head"]["location"]["z"] * scale

        rx = frame["head"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["head"]["rotation"]["y"] * rot_scale
        rz = frame["head"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        hmd_pose = self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
            0, 0, 0
        )

        def _get_button(button):
            if button:
                return 1
            else:
                return 0

        lx = frame["arm_r"]["location"]["x"] * scale
        ly = frame["arm_r"]["location"]["y"] * scale
        lz = frame["arm_r"]["location"]["z"] * scale

        rx = frame["arm_r"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["arm_r"]["rotation"]["y"] * rot_scale
        rz = frame["arm_r"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.CONTROLLER_t.pack(
            lx, ly, lz,  # x y z
            x, y, z, w,  # orientation quaternion
            float(velocity < 10), 0, 0,  # velocity
            0, 0, 0,  # angular velocity
            0, 0, _get_button(app_menu), 0, 0, 0, 0, 0, _get_button(click_trigger)  # controller inputs
        )

        lx = frame["arm_l"]["location"]["x"] * scale
        ly = frame["arm_l"]["location"]["y"] * scale
        lz = frame["arm_l"]["location"]["z"] * scale

        rx = frame["arm_l"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["arm_l"]["rotation"]["y"] * rot_scale
        rz = frame["arm_l"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.CONTROLLER_t.pack(
            lx, ly, lz,  # x y z
            x, y, z, w,  # orientation quaternion
            float(velocity < 10), 0, 0,  # velocity
            0, 0, 0,  # angular velocity
            0, 0, _get_button(app_menu), 0, 0, 0, 0, 0, _get_button(click_trigger)  # controller inputs
        )

        lx = frame["chest"]["location"]["x"] * scale
        ly = frame["chest"]["location"]["y"] * scale
        lz = frame["chest"]["location"]["z"] * scale

        rx = frame["chest"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["chest"]["rotation"]["y"] * rot_scale
        rz = frame["chest"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
            0, 0, 0
        )

        lx = frame["hip"]["location"]["x"] * scale
        ly = frame["hip"]["location"]["y"] * scale
        lz = frame["hip"]["location"]["z"] * scale

        rx = frame["hip"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["hip"]["rotation"]["y"] * rot_scale
        rz = frame["hip"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
            0, 0, 0
        )

        lx = frame["leg_l"]["location"]["x"] * scale
        ly = frame["leg_l"]["location"]["y"] * scale
        lz = frame["leg_l"]["location"]["z"] * scale

        rx = frame["leg_l"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["leg_l"]["rotation"]["y"] * rot_scale
        rz = frame["leg_l"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
            0, 0, 0
        )

        lx = frame["leg_r"]["location"]["x"] * scale
        ly = frame["leg_r"]["location"]["y"] * scale
        lz = frame["leg_r"]["location"]["z"] * scale

        rx = frame["leg_r"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["leg_r"]["rotation"]["y"] * rot_scale
        rz = frame["leg_r"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
            0, 0, 0
        )

        lx = frame["foot_l"]["location"]["x"] * scale
        ly = frame["foot_l"]["location"]["y"] * scale
        lz = frame["foot_l"]["location"]["z"] * scale

        rx = frame["foot_l"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["foot_l"]["rotation"]["y"] * rot_scale
        rz = frame["foot_l"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
            0, 0, 0
        )

        lx = frame["foot_r"]["location"]["x"] * scale
        ly = frame["foot_r"]["location"]["y"] * scale
        lz = frame["foot_r"]["location"]["z"] * scale

        rx = frame["foot_r"]["rotation"]["x"] * rot_scale
        ry = 1 - frame["foot_r"]["rotation"]["y"] * rot_scale
        rz = frame["foot_r"]["rotation"]["z"] * rot_scale

        rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
        x, y, z, w = rot.as_quat(False)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
            0, 0, 0
        )

        self.tracking_socket.sendall(hmd_pose + packet + self.SEND_TERMINATOR)

    @staticmethod
    def get_quaternion_from_euler(roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.

        Input
          :param roll: The roll (rotation around x-axis) angle in radians.
          :param pitch: The pitch (rotation around y-axis) angle in radians.
          :param yaw: The yaw (rotation around z-axis) angle in radians.

        Output
          :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)

        return [qx, qy, qz, qw]


class VRChatTrackingManager(SteamVRDeviceManager):
    def __init__(self):
        super().__init__()

        self.t_pose = {
                "head": {
                    "location": {
                        "x": 0.01643186155706644,
                        "y": 1.5838544368743896,
                        "z": 0.006781646981835365
                    },
                    "rotation": {
                        "x": -4.45107150420602,
                        "y": 0.3023844874613708,
                        "z": 177.82366261427208
                    },
                    "qrotation": [
                        -0.6543640913952851,
                        0.4289871317756008,
                        0.443472951881254,
                        0.4371606313623097
                    ]
                },
                "chest": {
                    "location": {
                        "x": 0.013884163869079202,
                        "y": 1.4058090448379517,
                        "z": -0.03805019520223141
                    },
                    "rotation": {
                        "x": -94.45107150420603,
                        "y": 0.3023844874613708,
                        "z": 177.82366261427208
                    },
                    "qrotation": [
                        -0.00189885605711946,
                        0.038876096759521514,
                        0.9990600749495566,
                        0.01907904840932058
                    ]
                },
                "hip": {
                    "location": {
                        "x": 0.012950003147125244,
                        "y": 0.9718396663665771,
                        "z": -0.021865202113986015
                    },
                    "rotation": {
                        "x": 262.78708513070364,
                        "y": -0.9360177182795297,
                        "z": -177.6372131709351
                    },
                    "qrotation": [
                        0.02006253156808301,
                        -0.9977848043598498,
                        -0.06271945458961378,
                        0.009447170562506654
                    ]
                },
                "leg_r": {
                    "location": {
                        "x": 0.1174139603972435,
                        "y": 0.49832937121391296,
                        "z": -0.06753223203122616
                    },
                    "rotation": {
                        "x": 76.25493240810223,
                        "y": -30.136775408717718,
                        "z": 164.26376512053974
                    },
                    "qrotation": [
                        0.03097711151505632,
                        0.5572958247142278,
                        0.046517383487075334,
                        0.8284309961317169
                    ]
                },
                "leg_l": {
                    "location": {
                        "x": -0.09790284931659698,
                        "y": 0.4971465468406677,
                        "z": -0.062113529071211815
                    },
                    "rotation": {
                        "x": -74.69178821006828,
                        "y": -23.00359695183448,
                        "z": 16.593248146304393
                    },
                    "qrotation": [
                        0.058637684223251635,
                        0.8604682672093071,
                        0.034910138807235716,
                        0.5049131265110922
                    ]
                },
                "foot_r": {
                    "location": {
                        "x": 0.15509212017059326,
                        "y": 0.017195334658026695,
                        "z": -0.0009366478770971298
                    },
                    "rotation": {
                        "x": 15.308211789931716,
                        "y": 66.99640304816552,
                        "z": 16.593248146304393
                    },
                    "qrotation": [
                        0.03097711151505632,
                        0.5572958247142278,
                        0.046517383487075334,
                        0.8284309961317169
                    ]
                },
                "foot_l": {
                    "location": {
                        "x": -0.13045676052570343,
                        "y": 0.012697434984147549,
                        "z": 0.01855609379708767
                    },
                    "rotation": {
                        "x": 166.25493240810223,
                        "y": 59.86322459128228,
                        "z": 164.26376512053974
                    },
                    "qrotation": [
                        0.058637684223251635,
                        0.8604682672093071,
                        0.034910138807235716,
                        0.5049131265110922
                    ]
                },
                "arm_r": {
                    "location": {
                        "x": 0.7127973437309265,
                        "y": 1.5472499132156372,
                        "z": -0.02000478468835354
                    },
                    "rotation": {
                        "x": 166.25493240810223,
                        "y": -0.13677540871771754,
                        "z": 164.26376512053974
                    },
                    "qrotation": [
                        0.41156772270990566,
                        0.4529153315941942,
                        -0.6842346362919245,
                        0.39661401201768065
                    ]
                },
                "arm_l": {
                    "location": {
                        "x": -0.6772607564926147,
                        "y": 1.5645933151245117,
                        "z": 0.027033517137169838
                    },
                    "rotation": {
                        "x": 15.308211789931716,
                        "y": 176.99640304816552,
                        "z": 16.593248146304393
                    },
                    "qrotation": [
                        -0.6543640913952851,
                        0.4289871317756008,
                        0.443472951881254,
                        0.4371606313623097
                    ]
                }
            }

    @staticmethod
    def launch_steamvr():
        # launches vrchat instead, we want to keep most of the
        # functionality so instead of creating a new class we simply override this function
        subprocess.run("start steam://run/438100", shell=True)

    def begin_calibration(self, calibration_path):
        with open(calibration_path, encoding="utf8") as calibration_opener:
            calibration_opener = json.load(calibration_opener)

        for idx, frame in enumerate(calibration_opener):
            velocity = idx
            frame = frame[0]

            packet = b''

            lx = frame["head"]["location"]["x"]
            ly = frame["head"]["location"]["y"]
            lz = frame["head"]["location"]["z"]

            rx = frame["head"]["rotation"]["x"]
            ry = 1 - frame["head"]["rotation"]["y"]
            rz = frame["head"]["rotation"]["z"]

            rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
            x, y, z, w = rot.as_quat(False)

            hmd_pose = self.POSE_t.pack(
                lx, ly, lz,
                x, y, z, w,
                float(velocity < 10), 0, 0,
                0, 0, 0
            )

            def _get_button(button):
                if button:
                    return 1
                else:
                    return 0

            if frame["click"]["arm"] == "right":
                if idx == 1:
                    activate_menu = 1
                else:
                    activate_menu = 0

                lx = frame["arm_r"]["location"]["x"]
                ly = frame["arm_r"]["location"]["y"]
                lz = frame["arm_r"]["location"]["z"]

                rx = frame["arm_r"]["rotation"]["x"]
                ry = 1 - frame["arm_r"]["rotation"]["y"]
                rz = frame["arm_r"]["rotation"]["z"]

                rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
                x, y, z, w = rot.as_quat(False)

                packet += self.CONTROLLER_t.pack(
                    lx, ly, lz,  # x y z
                    x, y, z, w,  # orientation quaternion
                    float(velocity < 10), 0, 0,  # velocity
                    0, 0, 0,  # angular velocity
                    0, 0, activate_menu, 0, 0, 0, 0, 0, _get_button(frame["click"]["click"])
                )

                lx = frame["arm_l"]["location"]["x"]
                ly = frame["arm_l"]["location"]["y"]
                lz = frame["arm_l"]["location"]["z"]

                rx = frame["arm_l"]["rotation"]["x"]
                ry = 1 - frame["arm_l"]["rotation"]["y"]
                rz = frame["arm_l"]["rotation"]["z"]

                rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
                x, y, z, w = rot.as_quat(False)

                packet += self.CONTROLLER_t.pack(
                    lx, ly, lz,  # x y z
                    x, y, z, w,  # orientation quaternion
                    float(velocity < 10), 0, 0,  # velocity
                    0, 0, 0,  # angular velocity
                    0, 0, 0, 0, 0, 0, 0, 0,0  # controller inputs
                )

            else:
                if idx == 1:
                    activate_menu = 1
                else:
                    activate_menu = 0

                lx = frame["arm_r"]["location"]["x"]
                ly = frame["arm_r"]["location"]["y"]
                lz = frame["arm_r"]["location"]["z"]

                rx = frame["arm_r"]["rotation"]["x"]
                ry = 1 - frame["arm_r"]["rotation"]["y"]
                rz = frame["arm_r"]["rotation"]["z"]

                rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
                x, y, z, w = rot.as_quat(False)

                packet += self.CONTROLLER_t.pack(
                    lx, ly, lz,  # x y z
                    x, y, z, w,  # orientation quaternion
                    float(velocity < 10), 0, 0,  # velocity
                    0, 0, 0,  # angular velocity
                    0, 0, activate_menu, 0, 0, 0, 0, 0, 0  # controller inputs
                )

                lx = frame["arm_l"]["location"]["x"]
                ly = frame["arm_l"]["location"]["y"]
                lz = frame["arm_l"]["location"]["z"]

                rx = frame["arm_l"]["rotation"]["x"]
                ry = 1 - frame["arm_l"]["rotation"]["y"]
                rz = frame["arm_l"]["rotation"]["z"]

                rot = R.from_euler('zyx', [rx, ry, rz], degrees=True)
                x, y, z, w = rot.as_quat(False)

                packet += self.CONTROLLER_t.pack(
                    lx, ly, lz,  # x y z
                    x, y, z, w,  # orientation quaternion
                    float(velocity < 10), 0, 0,  # velocity
                    0, 0, 0,  # angular velocity
                    0, 0, 0, 0, 0, 0, 0, 0, _get_button(frame["click"]["click"])  # controller inputs
                )


            packet += self.POSE_t.pack(
                0, 0, 0,
                0, 0, 0, 0,
                float(velocity < 10), 0, 0,
                0, 0, 0
            )

            packet += self.POSE_t.pack(
                0, 0, 0,
                0, 0, 0, 0,
                float(velocity < 10), 0, 0,
                0, 0, 0
            )

            packet += self.POSE_t.pack(
                0, 0, 0,
                0, 0, 0, 0,
                float(velocity < 10), 0, 0,
                0, 0, 0
            )

            packet += self.POSE_t.pack(
                0, 0, 0,
                0, 0, 0, 0,
                float(velocity < 10), 0, 0,
                0, 0, 0
            )

            packet += self.POSE_t.pack(
                0, 0, 0,
                0, 0, 0, 0,
                float(velocity < 10), 0, 0,
                0, 0, 0
            )

            packet += self.POSE_t.pack(
                0, 0, 0,
                0, 0, 0, 0,
                float(velocity < 10), 0, 0,
                0, 0, 0
            )

            self.tracking_socket.sendall(hmd_pose + packet + self.SEND_TERMINATOR)
            time.sleep(1/25)  # 25 fps for animation

        # the rest can be handled by the default self.update_pose_full class
        for i in range(100):
            self.update_pose_full(self.t_pose, velocity=i)
            time.sleep(1/25)
        self.update_pose_full(self.t_pose, velocity=101, click_trigger=True)
        print("[CALIBRATED] Pose should now be calibrated for use!")

if __name__ == "__main__":
    force_tracker_cam(False)
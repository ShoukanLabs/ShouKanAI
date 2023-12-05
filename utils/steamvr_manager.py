import json
import subprocess
import struct
import socket
import time
import numpy as np


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


def launch_steamvr():
    subprocess.run("start steam://run/250820", shell=True)


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

    def start_listening(self):
        launch_steamvr()
        time.sleep(5)

        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind(('', 6969))
        self.serversocket.listen(2)  # driver connects with 2 sockets

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
            return

        self.device_list = self.MANAGER_UDU_MSG_t.pack(
            20,  # HobovrManagerMsgType::Emsg_uduString
            9,  # 6 devices - 1 hmd, 2 controllers, 6 trackers
            0, 13,  # device description
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

    def update_pose_full(self, frame, app_menu=False, click_trigger=False, velocity=1, scale=2):

        packet = b''

        lx = frame["head"]["location"]["x"] * scale
        ly = frame["head"]["location"]["y"] * scale
        lz = frame["head"]["location"]["z"] * scale

        rx = frame["head"]["rotation"]["x"]
        ry = frame["head"]["rotation"]["y"]
        rz = frame["head"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        hmd_pose = self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            velocity / 10 / lx, ly, lz,
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

        rx = frame["arm_r"]["rotation"]["x"]
        ry = frame["arm_r"]["rotation"]["y"]
        rz = frame["arm_r"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.CONTROLLER_t.pack(
            lx, ly, lz,  # x y z
            x, y, z, w,  # orientation quaternion
            velocity / 10 / lx, ly, lz,  # velocity
            0, 0, 0,  # angular velocity
            0, 0, _get_button(app_menu), 0, 0, 0, 0, 0, _get_button(click_trigger)  # controller inputs
        )

        lx = frame["arm_l"]["location"]["x"] * scale
        ly = frame["arm_l"]["location"]["y"] * scale
        lz = frame["arm_l"]["location"]["z"] * scale

        rx = frame["arm_l"]["rotation"]["x"]
        ry = frame["arm_l"]["rotation"]["y"]
        rz = frame["arm_l"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.CONTROLLER_t.pack(
            lx, ly, lz,  # x y z
            x, y, z, w,  # orientation quaternion
            velocity / 10 / lx, ly, lz,  # velocity
            0, 0, 0,  # angular velocity
            0, 0, _get_button(app_menu), 0, 0, 0, 0, 0, _get_button(click_trigger)  # controller inputs
        )

        lx = frame["chest"]["location"]["x"] * scale
        ly = frame["chest"]["location"]["y"] * scale
        lz = frame["chest"]["location"]["z"] * scale

        rx = frame["chest"]["rotation"]["x"]
        ry = frame["chest"]["rotation"]["y"]
        rz = frame["chest"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            velocity / 10 / lx, ly, lz,
            0, 0, 0
        )

        lx = frame["hip"]["location"]["x"] * scale
        ly = frame["hip"]["location"]["y"] * scale
        lz = frame["hip"]["location"]["z"] * scale

        rx = frame["hip"]["rotation"]["x"]
        ry = frame["hip"]["rotation"]["y"]
        rz = frame["hip"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            velocity / 10 / lx, ly, lz,
            0, 0, 0
        )

        lx = frame["leg_l"]["location"]["x"] * scale
        ly = frame["leg_l"]["location"]["y"] * scale
        lz = frame["leg_l"]["location"]["z"] * scale

        rx = frame["leg_l"]["rotation"]["x"]
        ry = frame["leg_l"]["rotation"]["y"]
        rz = frame["leg_l"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            velocity / 10 / lx, ly, lz,
            0, 0, 0
        )

        lx = frame["leg_r"]["location"]["x"] * scale
        ly = frame["leg_r"]["location"]["y"] * scale
        lz = frame["leg_r"]["location"]["z"] * scale

        rx = frame["leg_r"]["rotation"]["x"]
        ry = frame["leg_r"]["rotation"]["y"]
        rz = frame["leg_r"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            velocity / 10 / lx, ly, lz,
            0, 0, 0
        )

        lx = frame["foot_l"]["location"]["x"] * scale
        ly = frame["foot_l"]["location"]["y"] * scale
        lz = frame["foot_l"]["location"]["z"] * scale

        rx = frame["foot_l"]["rotation"]["x"]
        ry = frame["foot_l"]["rotation"]["y"]
        rz = frame["foot_l"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            velocity / 10 / lx, ly, lz,
            0, 0, 0
        )

        lx = frame["foot_r"]["location"]["x"] * scale
        ly = frame["foot_r"]["location"]["y"] * scale
        lz = frame["foot_r"]["location"]["z"] * scale

        rx = frame["foot_r"]["rotation"]["x"]
        ry = frame["foot_r"]["rotation"]["y"]
        rz = frame["foot_r"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            velocity / 10 / lx, ly, lz,
            0, 0, 0
        )

        self.tracking_socket.sendall(hmd_pose + packet + self.SEND_TERMINATOR)
        print(f"[SENT]: {hmd_pose + packet + self.SEND_TERMINATOR}")


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
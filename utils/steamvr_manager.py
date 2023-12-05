import json
import subprocess
import struct
import socket
import time
import numpy as np
import pyrr


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

        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind(('', 6969))
        self.serversocket.listen(2)  # driver connects with 2 sockets

        print("[DRIVER] Waiting for driver to connect...")

        launch_steamvr()

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

    def create_trackers(self):
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

    def update_pose_full(self, frame, app_menu=False, click_trigger=False, velocity=1, scale=1):
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

        rx = frame["arm_r"]["rotation"]["x"]
        ry = frame["arm_r"]["rotation"]["y"]
        rz = frame["arm_r"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

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

        rx = frame["arm_l"]["rotation"]["x"]
        ry = frame["arm_l"]["rotation"]["y"]
        rz = frame["arm_l"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

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

        rx = frame["chest"]["rotation"]["x"]
        ry = frame["chest"]["rotation"]["y"]
        rz = frame["chest"]["rotation"]["z"]

        x, y, z, w = self.get_quaternion_from_euler(rx, ry, rz)

        packet += self.POSE_t.pack(
            lx, ly, lz,
            x, y, z, w,
            float(velocity < 10), 0, 0,
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
            float(velocity < 10), 0, 0,
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
            float(velocity < 10), 0, 0,
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
            float(velocity < 10), 0, 0,
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
            float(velocity < 10), 0, 0,
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


if __name__ == "__main__":
    TERMINATOR = b'\n'
    SEND_TERMINATOR = b'\t\r\n'
    MANAGER_UDU_MSG_t = struct.Struct("130I")
    POSE_t = struct.Struct("13f")
    CONTOLLER_t = struct.Struct("22f")

    # here we assume that no server was running and only our driver will connect
    # we assume the role of a server and a poser at the same time

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('', 6969))
    serversocket.listen(2)  # driver connects with 2 sockets

    #######################################################################
    # now lets accept both of them and resolve

    print("waiting for driver to connect...")

    client_a = serversocket.accept()
    client_b = serversocket.accept()

    print("waiting for driver resolution...")

    resp_a = client_a[0].recv(50)
    resp_b = client_b[0].recv(50)

    if TERMINATOR in resp_a:
        id_msg_a, resp_a = resp_a.split(TERMINATOR, 1)

    if TERMINATOR in resp_b:
        id_msg_b, resp_b = resp_b.split(TERMINATOR, 1)

    if id_msg_a == b"hello" and id_msg_b == b"monky":
        tracking_socket = client_a[0]
        manager_socket = client_b[0]

    elif id_msg_b == b"hello" and id_msg_a == b"monky":
        tracking_socket = client_b[0]
        manager_socket = client_a[0]

    else:
        print("bad connection")
        client_a[0].close()
        client_b[0].close()

        serversocket.close()

        exit()

    input("press anything to start the test...")

    #######################################################################
    # now lets add some trackers

    # tell the manager about current device setup
    device_list = MANAGER_UDU_MSG_t.pack(
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

    print(device_list)

    manager_socket.sendall(device_list + SEND_TERMINATOR)

    print("hmd with controllers and trackers: orbit...")

    print("press ctrl+C to stop...")

    try:
        i = 0

        while 1:
            q = pyrr.Quaternion.from_y_rotation(i / 180 * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            loc = np.array([0, -0.5, -1])

            # loc = mm.dot(loc)

            packet = b''

            temp = CONTOLLER_t.pack(
                *mm.dot(loc),
                1, 0, 0, 0,
                int(i < 10), 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0
            )
            packet += temp

            q = pyrr.Quaternion.from_y_rotation(i / (180 * 2) * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            temp = CONTOLLER_t.pack(
                *mm.dot(loc * [1, 1, 2]),
                1, 0, 0, 0,
                int(i < 10), 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0
            )
            packet += temp

            q = pyrr.Quaternion.from_y_rotation(i / (180 * 3) * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            temp = POSE_t.pack(
                *mm.dot(loc * [1, 1, 3]),
                1, 0, 0, 0,
                int(i < 10), 0, 0,
                0, 0, 0,
            )
            packet += temp

            q = pyrr.Quaternion.from_y_rotation(i / (180 * 3) * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            temp = POSE_t.pack(
                1, 1, 3,
                1, 3.2, 2.7362, 283.2827,
                int(i < 10), 0, 0,
                0, 0, 0,
            )
            packet += temp

            q = pyrr.Quaternion.from_y_rotation(i / (180 * 3) * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            temp = POSE_t.pack(
                *mm.dot(loc * [1, 1, 3]),
                1, 0, 0, 0,
                int(i < 10), 0, 0,
                0, 0, 0,
            )
            packet += temp

            q = pyrr.Quaternion.from_y_rotation(i / (180 * 3) * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            temp = POSE_t.pack(
                *mm.dot(loc * [1, 1, 3]),
                1, 0, 0, 0,
                int(i < 10), 0, 0,
                0, 0, 0,
            )
            packet += temp

            q = pyrr.Quaternion.from_y_rotation(i / (180 * 3.5) * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            temp = POSE_t.pack(
                *mm.dot(loc * [1, 1, 4]),
                1, 0, 0, 0,
                int(i < 10), 0, 0,
                0, 0, 0,
            )
            packet += temp

            q = pyrr.Quaternion.from_y_rotation(i / (180 * 3.8) * np.pi)

            mm = pyrr.matrix33.create_from_quaternion(q)

            temp = POSE_t.pack(
                *mm.dot(loc * [1, 1, 5]),
                1, 0, 0, 0,
                int(i < 10), 0, 0,
                0, 0, 0,
            )
            packet += temp

            hmd_pose = POSE_t.pack(
                int(i % 60 == 0) / 10, 0, 0,
                0, 0, -1, 0,
                # a trick to make steamvr think the hmd is active
                int(i % 60 == 0) / 10, 0, 0,
                0, 0, 0
            )

            tracking_socket.sendall(
                hmd_pose + packet + SEND_TERMINATOR
            )

            print(len(hmd_pose))
            print(len(packet))
            print(len(hmd_pose) + len(packet) + len(SEND_TERMINATOR))

            if i % 360 == 0:
                print(f"last q: {q}")

            time.sleep(1 / 60)
            i += 1

    except KeyboardInterrupt:
        print("interrupted, exiting...")

    #######################################################################
    # the end, time to die ^-^

    client_a[0].close()
    client_b[0].close()

    serversocket.close()
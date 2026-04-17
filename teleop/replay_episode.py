"""Replay recorded episode actions on the G1 robot with Inspire hands."""

import json
import sys
import time

import numpy as np

from robot_control.robot_arm import G1_29_ArmController
from robot_control.g1_robot_hand_inspire import InspireController

EPISODE_PATH = (
    "data/example/manipulation/fold_scarf/episode_2/data.json"
)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else EPISODE_PATH
    print(f"Loading {path}")
    with open(path, "r") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} frames")

    if not data_list or data_list[0]["actions"] is None:
        print("No actions to replay.")
        return

    arm_ctrl = G1_29_ArmController()
    hand_ctrl = InspireController()

    # Gradually ramp up arm weight (mirrors gradually_set_weight_to_0)
    print("Ramping up arm weight...")
    for w in range(1, 501):
        weight = w / 500
        arm_ctrl.msg.motor_cmd[29].q = weight
        arm_ctrl.msg.motor_cmd[30].q = weight
        arm_ctrl.msg.motor_cmd[31].q = weight
        arm_ctrl.msg.motor_cmd[32].q = weight
        arm_ctrl.msg.motor_cmd[33].q = weight
        arm_ctrl.msg.motor_cmd[34].q = weight
        time.sleep(0.01)
    print("Arm weight at 1.0, starting replay...")

    try:
        last_time = None
        for i in range(len(data_list)):
            actions = data_list[i]["actions"]
            if actions is None:
                continue

            # Timing: sleep to match original recording pace
            if last_time is not None:
                dt = data_list[i]["time"] - data_list[i - 1]["time"]
                elapsed = time.time() - last_time
                time.sleep(max(0, dt - elapsed))

            last_time = time.time()

            # Arm
            sol_q = actions["sol_q"]
            tau_ff = actions["tau_ff"]
            arm_ctrl.ctrl_dual_arm(sol_q, tau_ff)

            # Hands
            left_angles = actions["left_angles"]
            right_angles = actions["right_angles"]
            hand_ctrl.ctrl(left_angles, right_angles)

            if i % 30 == 0:
                print(f"  frame {i}/{len(data_list)}")

        print("Replay complete!")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Shutting down...")
        arm_ctrl.gradually_set_weight_to_0()
        arm_ctrl.shutdown()
        hand_ctrl.shutdown()


if __name__ == "__main__":
    main()

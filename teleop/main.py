import argparse

from manager import TeleopManager

# TODO: make it client server
# create a tv.step() thread and request image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Teleoperation Data Collector")
    parser.add_argument(
        "--task_name", type=str, default="default_task", help="Name of the task"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot",
        default="g1",
        choices=["g1", "h1", "g1_inspire"],
        help="Robot type: g1 (G1 + Dex3), h1 (H1 + Inspire), g1_inspire (G1 + Inspire FTP hands)",
    )

    # XR device selection
    parser.add_argument(
        "--xr-device",
        type=str,
        choices=["avp", "quest3"],
        default="avp",
        help="XR device type: avp (Apple Vision Pro) or quest3 (Meta Quest 3)",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["hand", "controller"],
        default="hand",
        help="Input tracking mode: hand (hand tracking) or controller (Quest controllers)",
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        choices=["immersive", "pass-through", "ego"],
        default="immersive",
        help="Display mode for XR headset",
    )

    args = parser.parse_args()

    manager = TeleopManager(
        task_name=args.task_name,
        robot=args.robot,
        debug=args.debug,
        xr_device=args.xr_device,
        input_mode=args.input_mode,
        display_mode=args.display_mode,
    )
    manager.start_processes()
    # TODO: run in two separate terminals for debuggnig
    manager.run_command_loop()

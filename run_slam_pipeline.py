"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('--slam_setting_file', type=str, default=None, help='Path to custom ORB-SLAM camera setting YAML (host path)')
@click.option('--camera_intrinsics', type=str, default=None, help='Path to camera intrinsics JSON file')
@click.option('--aruco_config', type=str, default=None, help='Path to ArUco config YAML file')
def main(session_dir, calibration_dir, slam_setting_file, camera_intrinsics, aruco_config):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()

    slam_setting_path = None
    if slam_setting_file is not None:
        slam_setting_path = pathlib.Path(os.path.expanduser(slam_setting_file)).absolute()
        assert slam_setting_path.is_file()

    if camera_intrinsics is None:
        camera_intrinsics_path = calibration_dir.joinpath('gopro_intrinsics_gopro13sn7674.json')
    else:
        camera_intrinsics_path = pathlib.Path(os.path.expanduser(camera_intrinsics)).absolute()

    if aruco_config is None:
        aruco_config_path = calibration_dir.joinpath('aruco_config.yaml')
    else:
        aruco_config_path = pathlib.Path(os.path.expanduser(aruco_config)).absolute()

    assert camera_intrinsics_path.is_file()
    assert aruco_config_path.is_file()

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############## 00_process_videos #############")
        script_path = script_dir.joinpath("00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 01_extract_gopro_imu ###########")
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_create_map ###########")
        script_path = script_dir.joinpath("02_create_map.py")
        assert script_path.is_file()
        demo_dir = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        print("mapping_dir", mapping_dir)
        assert mapping_dir.is_dir()
        map_path = mapping_dir.joinpath('map_atlas.osa')
        print("map_path", map_path)
        if not map_path.is_file():
            cmd = [
                'python', str(script_path),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path)
            ]
            if slam_setting_path is not None:
                cmd.extend([
                    '--setting_file', str(slam_setting_path)
                ])
            result = subprocess.run(cmd)
            assert result.returncode == 0
            assert map_path.is_file()

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path)
        ]
        if slam_setting_path is not None:
            cmd.extend([
                '--setting_file', str(slam_setting_path)
            ])
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04_detect_aruco ###########")
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()

        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--camera_intrinsics', str(camera_intrinsics_path),
            '--aruco_yaml', str(aruco_config_path)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 05_run_calibrations ###########")
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

## %%
if __name__ == "__main__":
    main()

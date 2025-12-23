"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2
from umi.common.cv_util import draw_predefined_mask

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('--setting_file', default="example/calibration/gopro13SN7674_maxlens_fisheye_setting_960x840.yaml", help='Host path to ORB-SLAM3 camera setting YAML file')
@click.option('-nm', '--no_mask', is_flag=True, default=True, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, docker_image, no_docker_pull, setting_file, no_mask):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    video_path = mount_target.joinpath('raw_video.mp4')
    json_path = mount_target.joinpath('imu_data.json')
    # mask_path = mount_target.joinpath('slam_mask.png')
    # if not no_mask:
    #     mask_write_path = video_dir.joinpath('slam_mask.png')
    #     slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
    #     slam_mask = draw_predefined_mask(
    #         slam_mask, color=255, mirror=True, gripper=False, finger=True)
    #     cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

    default_setting_path = '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml'
    setting_volume_args = []
    setting_path = default_setting_path
    if setting_file is not None:
        setting_file = pathlib.Path(os.path.expanduser(setting_file)).absolute()
        assert setting_file.is_file(), f"Setting file not found: {setting_file}"
        mount_target = pathlib.Path('/camera_settings')
        setting_volume_args = [
            '--volume', f"{setting_file.parent}:{mount_target}"
        ]
        setting_path = str(mount_target.joinpath(setting_file.name))

    # run SLAM
    cmd = [
        'docker',
        'run',
        '--rm', # delete after finish
        '--volume', str(video_dir) + ':' + '/data',
        '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
    ]
    cmd.extend(setting_volume_args)
    cmd.extend([
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', setting_path,
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ])
    # if not no_mask:
        # cmd.extend([
            # '--mask_img', str(mask_path)
        # ])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(result)


# %%
if __name__ == "__main__":
    main()

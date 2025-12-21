"""
python scripts_slam_pipeline/00_process_videos.py data_workspace/toss_objects/20231113
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
import shutil
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        # hardcode subdirs
        input_dir = session.joinpath('raw_videos')
        output_dir = session.joinpath('demos')
        motor_datas_dir = session.joinpath('motor_datas')
        gripper_cal_dir = session.joinpath('gripper_calibration')
        print(f"session: {session}")
        print(f"input_dir: {input_dir}")
        print(f"output_dir: {output_dir}")
        print(f"motor_datas_dir: {motor_datas_dir}")
        print(f"gripper_cal_dir: {gripper_cal_dir}")

        # create raw_videos if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)

        # create MP4 name map to imu json name
        mp4_name_to_imu_json_name = dict()
        mp4_name_to_motor_data_path = dict()
        mp4_name_to_motor_meta_data_path = dict()
        for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
            name_without_ext = mp4_path.with_suffix('').name
            imu_json_path = session.joinpath(name_without_ext + "_imu.json")
            motor_data_path = session.joinpath(name_without_ext + "_motor.npz")
            motor_meta_data_path = session.joinpath(name_without_ext + "_motor_meta.json")
            if imu_json_path.exists():
                out_json_path = input_dir.joinpath(name_without_ext + "_imu.json")
                shutil.move(imu_json_path, out_json_path)
                mp4_name_to_imu_json_name[name_without_ext] = out_json_path
            if motor_data_path.exists():
                out_motor_data_path = input_dir.joinpath(name_without_ext + "_motor.npz")
                shutil.move(motor_data_path, out_motor_data_path)
                mp4_name_to_motor_data_path[name_without_ext] = out_motor_data_path
            if motor_meta_data_path.exists():
                out_motor_meta_data_path = input_dir.joinpath(name_without_ext + "_motor_meta.json")
                shutil.move(motor_meta_data_path, out_motor_meta_data_path)
                mp4_name_to_motor_meta_data_path[name_without_ext] = out_motor_meta_data_path

        # create mapping video if don't exist
        mapping_vid_path = input_dir.joinpath('mapping.mp4')
        if (not mapping_vid_path.exists()) and not(mapping_vid_path.is_symlink()):
            max_size = -1
            max_path = None
            for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                size = mp4_path.stat().st_size
                if size > max_size:
                    max_size = size
                    max_path = mp4_path

            print(f"max_path: {max_path}, mapping_vid_path: {mapping_vid_path}")
            shutil.move(max_path, mapping_vid_path)
            imu_json_path = mp4_name_to_imu_json_name.get(max_path.with_suffix('').name, None)
            if imu_json_path is not None:
                shutil.move(imu_json_path, input_dir.joinpath('mapping_imu.json'))
            motor_data_path = mp4_name_to_motor_data_path.get(max_path.with_suffix('').name, None)
            if motor_data_path is not None:
                shutil.move(motor_data_path, input_dir.joinpath('mapping_motor.npz'))
            motor_meta_data_path = mp4_name_to_motor_meta_data_path.get(max_path.with_suffix('').name, None)
            if motor_meta_data_path is not None:
                shutil.move(motor_meta_data_path, input_dir.joinpath('mapping_motor_meta.json'))
            print(f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}.")
        # create gripper calibration video if don't exist
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial.")
            
            serial_start_dict = dict()
            serial_path_dict = dict()
            with ExifToolHelper() as et:
                for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                    if mp4_path.name.startswith('map'):
                        continue
                    
                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta['QuickTime:CameraSerialNumber']
                    
                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path
            
            for serial, path in serial_path_dict.items():
                print(f"Selected {path.name} for camera serial {serial}")
                out_path = gripper_cal_dir.joinpath(path.name)
                shutil.move(path, out_path)
                imu_path = mp4_name_to_imu_json_name.get(path.with_suffix('').name, None)
                if imu_path is not None:
                    imu_fname = "imu_data.json"
                    imu_out_path = gripper_cal_dir.joinpath(imu_fname)
                    shutil.move(imu_path, imu_out_path)
                motor_data_path = mp4_name_to_motor_data_path.get(path.with_suffix('').name, None)
                if motor_data_path is not None:
                    motor_fname = "motor.npz"
                    motor_out_path = gripper_cal_dir.joinpath(motor_fname)
                    shutil.move(motor_data_path, motor_out_path)
                motor_meta_data_path = mp4_name_to_motor_meta_data_path.get(path.with_suffix('').name, None)
                if motor_meta_data_path is not None:
                    motor_meta_fname = "motor_meta_data.json"
                    motor_meta_out_path = gripper_cal_dir.joinpath(motor_meta_fname)
                    shutil.move(motor_meta_data_path, motor_meta_out_path)

        # look for mp4 video in all subdirectories in input_dir
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # special folders
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"

                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # move videos
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # move imu jsons
                if out_dname == "mapping":
                    imu_path = input_dir.joinpath("mapping_imu.json")
                    out_imu_path = this_out_dir.joinpath("imu_data.json")
                    shutil.move(imu_path, out_imu_path)
                    motor_data_path = input_dir.joinpath("mapping_motor.npz")
                    out_motor_data_path = this_out_dir.joinpath("motor_data.npz")
                    shutil.move(motor_data_path, out_motor_data_path)
                    motor_meta_data_path = input_dir.joinpath("mapping_motor_meta.json")
                    out_motor_meta_data_path = this_out_dir.joinpath("motor_meta_data.json")
                    shutil.move(motor_meta_data_path, out_motor_meta_data_path)
                else:
                    imu_path = mp4_name_to_imu_json_name.get(mp4_path.with_suffix('').name, None)
                    if imu_path is not None:
                        out_imu_path = this_out_dir.joinpath("imu_data.json")
                        shutil.move(imu_path, out_imu_path)
                    motor_data_path = mp4_name_to_motor_data_path.get(mp4_path.with_suffix('').name, None)
                    if motor_data_path is not None:
                        out_motor_data_path = this_out_dir.joinpath("motor_data.npz")
                        shutil.move(motor_data_path, out_motor_data_path)
                    motor_meta_data_path = mp4_name_to_motor_meta_data_path.get(mp4_path.with_suffix('').name, None)
                    if motor_meta_data_path is not None:
                        out_motor_meta_data_path = this_out_dir.joinpath("motor_meta_data.json")
                        shutil.move(motor_meta_data_path, out_motor_meta_data_path)
                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()

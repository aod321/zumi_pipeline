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
        
        # create raw_videos if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)
        
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
            shutil.move(max_path, mapping_vid_path)
            print(f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}.")
        
        # create motor_datas if don't exist
        motor_datas_dir = session.joinpath('motor_datas')
        if not motor_datas_dir.is_dir():
            motor_datas_dir.mkdir()

        # move all motor files to motor_datas
        motor_files = list(session.glob('**/*_motor.npz')) + list(session.glob('**/*_motor_meta.json'))
        for motor_file in motor_files:
            if motor_file.parent == motor_datas_dir:
                continue
            dest_path = motor_datas_dir.joinpath(motor_file.name)
            if not dest_path.exists():
                shutil.move(motor_file, dest_path)
                print(f"Moved {motor_file.name} to motor_datas/")
         

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

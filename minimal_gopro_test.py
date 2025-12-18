import subprocess
import re
import sys
import os
import requests
import time
import click

def discover_gopro_ip():
    print("Discovering GoPro...")
    try:
        # Get all interfaces in oneline format, filtering out down/loopback/no-carrier
        cmd = "ip -4 --oneline link | grep -v 'state DOWN' | grep -v LOOPBACK | grep -v 'NO-CARRIER'"
        # Using shell=True to support pipes
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        lines = [line for line in output.strip().split("\n") if line.strip()]
            
        if not lines:
            print("No suitable interface found.")
            print("Camera not connected")
            return None
            
        # Iterate over all found interfaces to find one matching the GoPro IP pattern
        for line in lines:
            # Format is usually: "ID: NAME: <FLAGS> ..."
            parts = line.split(":")
            if len(parts) < 2:
                continue
                
            dev = parts[1].strip()
            
            # Get IP for this interface
            try:
                cmd_ip = f"ip -4 addr show dev {dev}"
                output_ip = subprocess.check_output(cmd_ip, shell=True).decode("utf-8")
            except Exception:
                continue
            
            # Regex to find IP
            match = re.search(r"inet ([\d\.]+)", output_ip)
            if not match:
                continue
                
            host_ip = match.group(1)
            
            # Construct GoPro IP (ends in .51)
            ip_parts = host_ip.split(".")
            if len(ip_parts) != 4:
                continue
                
            gopro_ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.51"
            
            # Validate against 172.2X.1YZ.51 rule
            # 172.2X.1YZ.51 where X, Y, Z are digits
            if re.match(r"^172\.2\d\.1\d\d\.51$", gopro_ip):
                print(f"Found interface: {dev}")
                print(f"Host IP on interface: {host_ip}")
                print(f"Target GoPro IP: {gopro_ip}")
                return gopro_ip

        # If loop finishes without success
        print("Camera not connected")
        return None

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return None
    except Exception as e:
        print(f"Discovery failed: {e}")
        return None

def get_ip_from_sn(sn):
    """
    Derive IP from Serial Number (SN).
    Logic: 172.2X.1YZ.51 where XYZ are the last three digits of the SN.
    """
    if len(sn) < 3:
        raise ValueError("Serial Number must be at least 3 characters long.")
    
    last_three = sn[-3:]
    if not last_three.isdigit():
        raise ValueError(f"Last 3 characters of SN must be digits. Got: {last_three}")
        
    x = last_three[0]
    y = last_three[1]
    z = last_three[2]
    
    # IP: 172.2X.1YZ.51
    return f"172.2{x}.1{y}{z}.51"

class GoPro:
    def __init__(self, ip):
        self.base_url = f"http://{ip}:8080"
    
    def get_info(self):
        """Get camera info including Serial Number."""
        try:
            r = requests.get(f"{self.base_url}/gopro/camera/info", timeout=2)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Error getting camera info: {e}")
            return None

    def get_state(self):
        """Get camera status."""
        try:
            r = requests.get(f"{self.base_url}/gopro/camera/state", timeout=2)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Error getting camera state: {e}")
            return

    def enable_wired_control(self):
        print("Enabling Wired USB Control...")
        try:
            # /gopro/camera/control/wired_usb?p=1
            r = requests.get(f"{self.base_url}/gopro/camera/control/wired_usb?p=1", timeout=5)
            r.raise_for_status()
            print("Wired USB Control enabled successfully.")
        except Exception as e:
            # Often this fails if already enabled or not supported, but good to try.
            # Don't print stack trace to keep output clean, just the error message.
            print(f"Note: Enabling Wired USB Control returned: {e}")

    def start_recording(self):
        print("Sending Start Recording command...")
        
        # Proactively enable wired control
        # self.enable_wired_control()
        
        try:
            # /gopro/camera/shutter/start
            r = requests.get(f"{self.base_url}/gopro/camera/shutter/start", timeout=5)
            r.raise_for_status()
            print("Response:", r.text)
            print("Recording started successfully.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500:
                print("Error 500: Internal Server Error.")
                print("Likely causes: No SD card, SD card full, camera busy, or wrong mode.")
                print("Attempting to set Video mode (Group 1000) and retry...")
                
                # Wait for mode switch to settle
                time.sleep(2)
                
                try:
                     r = requests.get(f"{self.base_url}/gopro/camera/shutter/start", timeout=5)
                     r.raise_for_status()
                     print("Retry successful! Recording started.")
                     return
                except Exception as retry_e:
                     print(f"Retry failed: {retry_e}")

                print("Checking camera state...")
                state = self.get_state()
                if state:
                    # Print full state for debugging
                    print(f"Camera State: {state}")
            else:
                print(f"Error starting recording: {e}")
        except Exception as e:
            print(f"Error starting recording: {e}")

    def stop_recording(self):
        print("Sending Stop Recording command...")
        try:
            # /gopro/camera/shutter/stop
            r = requests.get(f"{self.base_url}/gopro/camera/shutter/stop", timeout=5)
            r.raise_for_status()
            print("Response:", r.text)
            print("Recording stopped successfully.")
        except Exception as e:
            print(f"Error stopping recording: {e}")

    def download_last_video(self):
        print("Checking last captured media...")
        try:
            # Get last media info
            r = requests.get(f"{self.base_url}/gopro/media/last_captured", timeout=5)
            r.raise_for_status()
            data = r.json()
            
            directory = data.get('folder')
            filename = data.get('file')
            
            if not directory or not filename:
                print("Could not retrieve valid media info.")
                return

            print(f"Last media found: {directory}/{filename}")
            
            if not filename.lower().endswith('.mp4'):
                print("Last captured file is not an MP4. Skipping download.")
                return

            # Download
            download_url = f"{self.base_url}/videos/DCIM/{directory}/{filename}"
            print(f"Downloading from {download_url}...")
            
            with requests.get(download_url, stream=True) as r_file:
                r_file.raise_for_status()
                total_size = int(r_file.headers.get('content-length', 0))
                
                with open(filename, 'wb') as f:
                    downloaded = 0
                    for chunk in r_file.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                sys.stdout.write(f"\rProgress: {percent:.1f}%")
                                sys.stdout.flush()
            
            print(f"\nDownload complete! Saved to {filename}")
            
        except Exception as e:
            print(f"\nError downloading media: {e}")

@click.command()
@click.option('--sn', help='GoPro Serial Number (e.g., C3461325434789). Calculates IP automatically.')
@click.option('--ip', help='GoPro IP Address (e.g., 172.24.189.51). If provided, attempts to fetch SN from camera.')
def main(sn, ip):
    print("--- Minimal GoPro Test Script ---")
    
    target_ip = None

    if sn:
        print(f"Input Serial Number: {sn}")
        try:
            target_ip = get_ip_from_sn(sn)
            print(f"Calculated IP from SN: {target_ip}")
        except ValueError as e:
            print(f"Error deriving IP from SN: {e}")
            sys.exit(1)
    elif ip:
        print(f"Input IP Address: {ip}")
        target_ip = ip
    else:
        # Auto Discovery
        target_ip = discover_gopro_ip()
        if not target_ip:
            sys.exit(1)

    # Initialize Camera
    cam = GoPro(target_ip)

    # If IP was provided manually (or auto-discovered), let's try to get/print the SN
    # Note: If SN was provided, we already know it (or at least the user thinks they know it), 
    # but we can still verify connection.
    print(f"Connecting to {target_ip}...")
    
    # Proactively enable wired control at startup to avoid delay later
    cam.enable_wired_control()
    
    info = cam.get_info()
    
    if info:
        print(f"Debug Info: {info}")
        camera_sn = info.get('serial_number')
        if not camera_sn:
             camera_sn = info.get('info', {}).get('serial_number')
        print(f"Successfully connected! Camera SN: {camera_sn}")
        if sn and camera_sn != sn:
             print(f"Warning: Provided SN ({sn}) does not match reported Camera SN ({camera_sn})")
    else:
        print("Warning: Could not connect to camera to verify Serial Number or connection status.")
        if not sn and not ip:
             # If we failed on auto-discovery connection, maybe we shouldn't proceed?
             # But maybe the info endpoint is flaky? Let's assume we proceed if the user wants to try.
             pass

    # 2. Interactive Menu
    while True:
        print("\nOptions:")
        print("1. Start Recording")
        print("2. Stop Recording")
        print("3. Download Latest MP4")
        print("q. Quit")
        
        choice = input("Enter choice: ").strip()
        
        if choice == '1':
            cam.start_recording()
        elif choice == '2':
            cam.stop_recording()
        elif choice == '3':
            cam.download_last_video()
        elif choice == 'q':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()

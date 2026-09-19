import os
import zipfile
import subprocess
import shutil
from pathlib import Path

# Paths
DATASET_ROOT = Path("Dataset/VidHOI")
EXTRACT_DIR = DATASET_ROOT / "extracted"

ZIPS_TO_EXTRACT = [
    "validation-annotation.zip",
    "training-annotation.zip",
    "validation-video.zip"
]

def check_ffmpeg():
    """Checks if ffmpeg is in the system PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("Found ffmpeg on system PATH.")
        return "ffmpeg"
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not found on system PATH.")
        return None

def setup_ffmpeg():
    """Extracts bundled ffmpeg if needed and returns the executable path."""
    ffmpeg_exe = check_ffmpeg()
    if ffmpeg_exe:
        return ffmpeg_exe

    ffmpeg_zip = DATASET_ROOT / "ffmpeg-3.3.4.zip"
    if not ffmpeg_zip.exists():
        raise FileNotFoundError(f"ffmpeg not on PATH and bundled zip not found at {ffmpeg_zip}")
    
    ffmpeg_extract_dir = DATASET_ROOT / "ffmpeg"
    if not ffmpeg_extract_dir.exists():
        print(f"Extracting {ffmpeg_zip}...")
        with zipfile.ZipFile(ffmpeg_zip, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_extract_dir)
            
    # Find the ffmpeg.exe inside the extracted folder
    for root, dirs, files in os.walk(ffmpeg_extract_dir):
        if "ffmpeg.exe" in files:
            exe_path = os.path.join(root, "ffmpeg.exe")
            print(f"Found bundled ffmpeg at: {exe_path}")
            return exe_path
            
    raise FileNotFoundError("Could not find ffmpeg.exe inside the extracted ffmpeg zip.")

def extract_datasets():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    for zip_name in ZIPS_TO_EXTRACT:
        zip_path = DATASET_ROOT / zip_name
        if not zip_path.exists():
            print(f"Warning: {zip_path} not found. Skipping.")
            continue
            
        print(f"Extracting {zip_path} to {EXTRACT_DIR}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
            print(f"Successfully extracted {zip_name}")
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is a bad zip file.")

def decode_videos(ffmpeg_path, fps=5):
    """Uses ffmpeg to decode videos into frames."""
    video_dir = EXTRACT_DIR / "video"  # Assuming videos are extracted here, adjust if needed
    frames_dir = EXTRACT_DIR / "frames"
    
    if not video_dir.exists():
         # Sometimes validation-video.zip extracts directly or into a different structure.
         # Let's search for mp4 files.
         print(f"Searching for videos in {EXTRACT_DIR}...")
         video_files = list(EXTRACT_DIR.rglob("*.mp4"))
    else:
         video_files = list(video_dir.rglob("*.mp4"))

    if not video_files:
        print("No .mp4 videos found to decode.")
        return

    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(video_files)} videos. Decoding at {fps} fps...")
    
    for i, video_path in enumerate(video_files):
        video_id = video_path.stem
        video_frame_dir = frames_dir / video_id
        
        if video_frame_dir.exists() and any(video_frame_dir.iterdir()):
             continue # Already decoded

        video_frame_dir.mkdir(parents=True, exist_ok=True)
        
        # ffmpeg -i input.mp4 -vf fps=5 %06d.jpg
        out_pattern = str(video_frame_dir / "%06d.jpg")
        cmd = [
            ffmpeg_path,
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2", # High quality jpeg
            out_pattern
        ]
        
        print(f"Decoding {video_id} ({i+1}/{len(video_files)})...", end='\r')
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nError decoding {video_id}: {e}")
            
    print("\nFinished decoding all videos.")

def main():
    print("Starting VidHOI Setup...")
    ffmpeg_path = setup_ffmpeg()
    extract_datasets()
    decode_videos(ffmpeg_path, fps=5)
    print("Setup complete.")

if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import shutil

def organize_images() -> None:
    base_dir = Path('dataset/images')

    party_list_dir = base_dir / "party_list"
    constituency_dir = base_dir / "constituency"

    party_list_dir.mkdir(parents=True, exist_ok=True)
    constituency_dir.mkdir(parents=True, exist_ok=True)

    for file_path in base_dir.glob('*.png'):
        filename = file_path.name
    
        if "party_list" in filename:
            target_path = party_list_dir / filename
            shutil.move(str(file_path), str(target_path))
            print(f"Moved: {filename} -> party_list/")
        
        elif "constituency" in filename:
            target_path = constituency_dir / filename
            shutil.move(str(file_path), str(target_path))



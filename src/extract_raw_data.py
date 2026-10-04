import os
import shutil
import zipfile

RAW_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\data\raw"
DS_ZIP = r"C:\Users\akhil\Downloads\Dataset.zip"
POLAR_SRC = r"C:\Users\akhil\Downloads\FFA_W3_241_polar.csv"
IEA_SRC = r"C:\Users\akhil\Downloads\IEA_Wind_Task_46_WP3.1_AEP_Loss-1.pdf"

print("Copying polar file...")
shutil.copy(POLAR_SRC, os.path.join(RAW_DIR, "FFA_W3_241_polar.csv"))

print("Copying IEA report...")
shutil.copy(IEA_SRC, os.path.join(RAW_DIR, "IEA_Wind_Task_46_WP3.1_AEP_Loss-1.pdf"))

img_dir = os.path.join(RAW_DIR, "images")
ann_dir = os.path.join(RAW_DIR, "annotations")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

print("Extracting Dataset.zip files...")
with zipfile.ZipFile(DS_ZIP, 'r') as z:
    for member in z.infolist():
        filename = member.filename
        if filename.endswith('T1.csv'):
            target_path = os.path.join(RAW_DIR, "T1.csv")
            with z.open(member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
            print("Extracted T1.csv")
        elif 'Annotations' in filename and filename.endswith('.xml'):
            base = os.path.basename(filename)
            if base:
                target_path = os.path.join(ann_dir, base)
                with z.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
        elif 'JPEGImages' in filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.basename(filename)
            if base:
                target_path = os.path.join(img_dir, base)
                with z.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

print(f"Extraction complete! Images: {len(os.listdir(img_dir))}, Annotations: {len(os.listdir(ann_dir))}")

import os
import subprocess
from multiprocessing import Pool
from huggingface_hub import hf_hub_download

def single_uncompress(file, dest="/mnt/data"):
    local_path = hf_hub_download(
        repo_id="OpenShape/openshape-training-data",
        filename=file,
        repo_type="dataset",
        local_dir=dest)
    
    try:
        if file.endswith(".tar.gz"):
            subprocess.run(["tar", "-xzf", local_path, "-C", dest], check=True)
        elif file.endswith(".zip"):
            subprocess.run(["unzip", local_path, "-d", dest], check=True)
        else:
            print(f"File extension not supported: {file}")
            return
        os.remove(local_path)
    except Exception as e:
        print(f"Erro ao extrair {file}: {e}")

NUM_PROC = 8  # reduzi para não sobrecarregar
pool = Pool(NUM_PROC)

# sempre passando o dest explicitamente
pool.apply_async(single_uncompress, ("meta_data.zip", "./"))

files = ["3D-FUTURE.tar.gz", "ABO.tar.gz", "ShapeNet.tar.gz"]
for file in files:
    pool.apply_async(single_uncompress, (file, "./"))

for i in range(160):
    pool.apply_async(single_uncompress, (f"Objaverse/000-{i:03d}.tar.gz", "./"))

pool.close()
pool.join()


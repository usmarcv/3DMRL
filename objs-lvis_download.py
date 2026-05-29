import argparse
import glob
import gzip
import json
import multiprocessing
import os
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

BASE_PATH = os.path.join("objaverse")
__version__ = "1.0"
DEFAULT_VERSIONED_PATH = os.path.join(BASE_PATH, "sneakers")


def load_annotations(uids: Optional[List[str]] = None, versioned_path: str = DEFAULT_VERSIONED_PATH) -> Dict[str, Any]:
    """Load the full metadata of all objects in the dataset."""
    metadata_path = os.path.join(versioned_path, "metadata")
    object_paths = _load_object_paths(versioned_path)
    dir_ids = (
        set([object_paths[uid].split("/")[1] for uid in uids])
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    if len(dir_ids) > 10:
        dir_ids = tqdm(dir_ids)
    out = {}
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.update(data)
        if uids is not None and len(out) == len(uids):
            break
    return out


def _load_object_paths(versioned_path: str) -> Dict[str, str]:
    """Load the object paths from the dataset."""
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(versioned_path, object_paths_file)
    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def load_uids(versioned_path: str = DEFAULT_VERSIONED_PATH) -> List[str]:
    """Load the uids from the dataset."""
    return list(_load_object_paths(versioned_path).keys())


def _download_object(
    uid: str,
    object_path: str,
    total_downloads: float,
    start_file_count: int,
    versioned_path: str,
    category: str = "Desconhecida",
) -> Tuple[str, str]:
    """Download the object for the given uid and print its category."""
    local_path = os.path.join(versioned_path, object_path)
    tmp_local_path = os.path.join(versioned_path, object_path + ".tmp")
    hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
    
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(hf_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)

    files = glob.glob(os.path.join(versioned_path, "glbs", "*", "*.glb"))
    
    print(
        f"Downloaded {len(files) - start_file_count} / {int(total_downloads)} objects | "
        f"UID: {uid} | Categoria: {category}"
    )

    return uid, local_path


def load_objects(
    uids: List[str], 
    download_processes: int = 1, 
    uid_to_category: Optional[Dict[str, str]] = None,
    versioned_path: str = DEFAULT_VERSIONED_PATH
) -> Dict[str, str]:
    """Return the path to the object files, mapping them to their categories during download."""
    object_paths = _load_object_paths(versioned_path)
    out = {}
    uid_to_category = uid_to_category or {}

    if download_processes == 1:
        uids_to_download = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(versioned_path, object_path)
            if os.path.exists(local_path):
                out[uid] = local_path
                continue
            uids_to_download.append((uid, object_path))
        
        if len(uids_to_download) == 0:
            return out
        
        start_file_count = len(glob.glob(os.path.join(versioned_path, "glbs", "*", "*.glb")))
        for uid, object_path in uids_to_download:
            cat = uid_to_category.get(uid, "Desconhecida")
            uid, local_path = _download_object(
                uid, object_path, len(uids_to_download), start_file_count, versioned_path, category=cat
            )
            out[uid] = local_path
    else:
        args = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(versioned_path, object_path)
            if not os.path.exists(local_path):
                cat = uid_to_category.get(uid, "Desconhecida")
                args.append((uid, object_paths[uid], cat))
            else:
                out[uid] = local_path
        
        if len(args) == 0:
            return out
            
        print(f"starting download of {len(args)} objects with {download_processes} processes")
        start_file_count = len(glob.glob(os.path.join(versioned_path, "glbs", "*", "*.glb")))
        
        # Ajusta a tupla de argumentos incluindo o versioned_path dinâmico
        args_list = [(arg[0], arg[1], len(args), start_file_count, versioned_path, arg[2]) for arg in args]
        
        with multiprocessing.Pool(download_processes) as pool:
            r = pool.starmap(_download_object, args_list)
            for uid, local_path in r:
                out[uid] = local_path
                
    return out


def load_lvis_annotations(versioned_path: str = DEFAULT_VERSIONED_PATH) -> Dict[str, List[str]]:
    """Load the LVIS annotations."""
    hf_url = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"
    local_path = os.path.join(versioned_path, "lvis-annotations.json.gz")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        lvis_annotations = json.load(f)
    return lvis_annotations


if __name__ == "__main__":
    # Configuração do Argparse para receber parâmetros por linha de comando
    parser = argparse.ArgumentParser(description="Download de objetos do Objaverse filtrados por categorias.")
    
    parser.add_argument(
        "--caminho_txt", 
        type=str, 
        default="retrieval_results/sneakers.txt", 
        help="Caminho para o arquivo TXT que contém a lista de UIDs."
    )
    
    parser.add_argument(
        "--versioned_path", 
        type=str, 
        default=DEFAULT_VERSIONED_PATH, 
        help="Diretório onde os arquivos baixados e metadados serão salvos."
    )
    
    args = parser.parse_args()

    # 1. Carregar os UIDs do TXT parametrizado
    if not os.path.exists(args.caminho_txt):
        print(f"Erro: O arquivo {args.caminho_txt} não foi encontrado.")
        exit()

    with open(args.caminho_txt, "r") as f:
        uids_especificos = [line.strip() for line in f if line.strip()]

    print(f"Carregados {len(uids_especificos)} UIDs do arquivo de texto: {args.caminho_txt}")

    # 2. Carregar as anotações LVIS
    print(f"Carregando mapeamento de categorias LVIS em: {args.versioned_path} ...")
    lvis_annotations = load_lvis_annotations(versioned_path=args.versioned_path)
    
    uid_to_category = {}
    for categoria, lista_uids in lvis_annotations.items():
        for uid in lista_uids:
            uid_to_category[uid] = categoria

    # 3. Mostrar resumo das categorias
    print("\n--- Resumo das Categorias Encontradas no Arquivo ---")
    contagem_categorias = {}
    for uid in uids_especificos:
        uid_limpo = uid[:-4] if uid.endswith(".glb") else uid
        cat = uid_to_category.get(uid_limpo, "Não listado no LVIS")
        contagem_categorias[cat] = contagem_categorias.get(cat, 0) + 1

    for cat, total in sorted(contagem_categorias.items(), key=lambda x: x[1], reverse=True):
        print(f"* {cat}: {total} objeto(s)")
    print("---------------------------------------------------\n")


    print("Iniciando o download dos objetos...")
    objects = load_objects(
        uids_especificos, 
        download_processes=10, 
        uid_to_category=uid_to_category, 
        versioned_path=args.versioned_path
    )

    
    print(f"\nProcesso concluído! Baixados {len(objects)} objetos salvos em '{args.versioned_path}'.")
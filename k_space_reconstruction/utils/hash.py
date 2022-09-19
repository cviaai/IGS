import os
import hashlib
from tqdm import tqdm


def update_hash_on_file(fp, hs):
    with open(fp, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hs.update(chunk)
    return True


def get_file_md5hash(fp):
    hs = hashlib.md5()
    update_hash_on_file(fp, hs)
    return hs.hexdigest()


def get_dir_md5hash(dp):
    hs = hashlib.md5()
    files = []
    for root, _, fs in os.walk(dp):
        files += [os.path.join(root, f) for f in fs]
    for f in tqdm(sorted(files)):
        update_hash_on_file(f, hs)
    return hs.hexdigest()

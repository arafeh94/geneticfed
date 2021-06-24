from src import manifest
from src.apis import files

maps = files.load(manifest.DEFAULT_ACC_PATH)
print(maps)

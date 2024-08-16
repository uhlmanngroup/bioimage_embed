# %%
import fsspec

# %%
# import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterableWrapper
from PIL import Image
import io
from joblib import Memory
from PIL import UnidentifiedImageError

# %%
memory = Memory(location=".", verbose=0)

# %%
# Define FTP host and root directory
host = "ftp.ebi.ac.uk"
root = "pub/databases/IDR"
dataset = "idr0093-mueller-perturbation"

# %%
# # Setup fsspec filesystem for FTP access
# fs = fsspec.filesystem("ftp", host=host, anon=True)
fs = fsspec.filesystem("ftp", host=host, anon=True)

# %% [markdown]
# # Glob pattern to match the files you're interested in

# %% [markdown]
# glob_str = f"{root}/{dataset}/**/"
# folders = fs.glob(glob_str, recursive=True)
# dp = IterableWrapper(folders).list_files_by_fsspec(
#     anon=True,
#     protocol="ftp",
#     host=host,
#     recursive=True,
#     masks=["*.tif", "*.tiff"],
# )

# %%
glob_str = f"{root}/{dataset}/**/*.tif*"


# %%
@memory.cache
def get_file_list(glob_str, fs):
    return fs.glob(glob_str, recursive=True)


# %%
files = get_file_list(glob_str, fs)


# %%
def read_file(x):
    try:
        # Attempt to open the image
        print(x[0])
        stream = x[1].read()
        print("Valid file")
        return stream
    except Exception:
        print("Invalid file")
        return None


# %%
def read_image(x):
    return Image.open(io.BytesIO(x))


# %%
def is_valid_image(x):
    try:
        # Attempt to open the image
        image = read_image(x)
        image.verify()  # Ensure it's a valid image
        print("Valid image")
        return True
    except (IOError, UnidentifiedImageError):
        print("Invalid image")
        return False


# %%
dp = (
    # IterableWrapper(files)
    IterableWrapper(files)
    .open_files_by_fsspec(
        anon=True,
        protocol="ftp",
        host=host,
        mode="rb",
        filecache={"cache_storage": "tmp/idr"},
    )
    # .filter(filter_fn=is_valid_file)
    .map(read_file)
    .filter(filter_fn=is_valid_image)
    .map(lambda x: Image.open(io.BytesIO(x)))
)

# %%
a = next(iter(dp))
print(a)

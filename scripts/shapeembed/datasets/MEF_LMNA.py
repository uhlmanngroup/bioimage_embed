import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
import torch
import torchvision

# here is where the dataset comes from:
# from https://zenodo.org/records/7388245#.Y4k10ezMJqs
# https://zenodo.org/records/7388245/files/mefs.tar.gz
# 
# how to get it:
# $ wget https://zenodo.org/records/7388245/files/mefs.tar.gz
# $ tar -xvzf mefs.tar.gz

# here are the default paths to the dataset we use if not otherwise specified
dfltpath_original=Path('/nfs/research/uhlmann/afoix/datasets/image_datasets/mefs')
dfltpath_single_obj=Path('/nfs/research/uhlmann/afoix/datasets/image_datasets/mefs_single_object_masks')

class MEF_LMNA_Base(torch.utils.data.Dataset):
  """
  A Dataset to capture the original MEF LMNA dataset with multiple objects per
  mask

  This class is meant to be an Abstract base class and should only be inherited
  from, providing at least an extra __init__ method

  This class provides a retrieve_samples helper static method as well as the
  pattern_classes and lmna_classes class variables

  This class also provides a __getitem__ and a __getlen__ implementation
  assuming a populated 'samples' instance variable. These method can be
  overridden if preferred.

  __getitem__ return a sample of the form: (path, (pattern_idx, lmna_idx))
  where path is the path to the image, pattern_idx is the index into the
  "pattern" class list ['Control', 'Triangle', 'Circle'] and lmna_idx is the
  index into the lmna class list ['LMNA++1', 'LMNA--1', 'LMNA++2', 'LMNA--2']
  """

  # the pattern classes, ordered
  pattern_classes = ['Control', 'Triangle', 'Circle']
  # the lmna classes, ordered
  lmna_classes = ['LMNA++1', 'LMNA--1', 'LMNA++2', 'LMNA--2']

  # helper function to retrieve and classify the samples
  @staticmethod
  def retrieve_samples(root_path, include_cells, include_nuclei):
    samples = []
    if include_cells: cell_re = re.compile('.*cell.*\.png')
    if include_nuclei: nuc_re = re.compile('.*nuc.*\.png')
    for i, pattern in enumerate(MEF_LMNA_Base.pattern_classes):
      for j, lmna in enumerate(MEF_LMNA_Base.lmna_classes):
        for f in (root_path / pattern / lmna).iterdir():
          if include_cells and cell_re.match(str(f)):
            samples.append((f, (i, j)))
          elif include_nuclei and nuc_re.match(str(f)):
            samples.append((f, (i, j)))
    return samples

  def __getitem__(self, index: int) -> Tuple[Path, Tuple[int, int]]:
    return self.samples[index]

  def __len__(self) -> int:
    return len(self.samples)

# Original dataset (all objects in masks)
################################################################################

class MEF_LMNA_Original(MEF_LMNA_Base):
  def __init__(self, path=dfltpath_original, include_cells=True, include_nuclei=True) -> None:
    # remember the dataset path
    self.root_path = Path(path)
    # retrieve and classify the samples
    self.samples = MEF_LMNA_Base.retrieve_samples(
                     self.root_path / 'data' / 'processed'
                   , include_cells
                   , include_nuclei )

class MEF_LMNA_Original_cells(MEF_LMNA_Original):
  def __init__(self, path=dfltpath_original):
    super().__init__(path=path, include_cells=True, include_nuclei=False)

class MEF_LMNA_Original_nuclei(MEF_LMNA_Original):
  def __init__(self, path=dfltpath_original):
    super().__init__(path=path, include_cells=False, include_nuclei=True)

class MEF_LMNA_Original_Image(MEF_LMNA_Original):
  def __init__(self, path=dfltpath_original, include_cells=False, include_nuclei=True):
    super().__init__( path=path
                    , include_cells = include_cells
                    , include_nuclei = include_nuclei )

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
      path, classes = super().__getitem__(index)
      return(torchvision.io.read_image(path), classes)

class MEF_LMNA_Original_Image_cells(MEF_LMNA_Original_Image):
  def __init__(self, path=dfltpath_original):
    super().__init__(path=path, include_cells=True, include_nuclei=False)

class MEF_LMNA_Original_Image_nuclei(MEF_LMNA_Original_Image):
  def __init__(self, path=dfltpath_original):
    super().__init__(path=path, include_cells=True, include_nuclei=False)

################################################################################

def MEF_LMNA_extract_single_object_masks( dataset: MEF_LMNA_Original
                                        , outpath: Path = None) -> None:
  """
  This function extracts goes through a dataset of multi object masks and
  extracts individual objects into their own mask
  """
  # where to generate the single object mask version of the dataset
  if not outpath:
    d = dataset.root_path
    outpath = d.with_name(f'{d.name}_single_object_masks')
  # create the destination folder structure as needed
  if not outpath.exists(): outpath.mkdir()
  for pat in dataset.pattern_classes:
    if not (outpath / pat).exists(): (outpath / pat).mkdir()
    for lmna in dataset.lmna_classes:
      if not (outpath / pat / lmna).exists(): (outpath / pat / lmna).mkdir()
  # helper bounding box function
  def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin, ymax, xmin, xmax
  # traverse the dataset and extract the individual objects
  for path, (patidx, lmnaidx) in tqdm(dataset):
    all_masks = torchvision.io.read_image(path).numpy()[0] # drop extra channel
    for i, lbl in enumerate(set(np.unique(all_masks)) - set({0})):
      this_mask = (all_masks == lbl).astype(np.uint8)
      ymin, ymax, xmin, xmax =  bbox(this_mask)
      this_mask = this_mask[ymin:ymax+1, xmin:xmax+1]
      patdir = dataset.pattern_classes[patidx]
      lmnadir = dataset.lmna_classes[lmnaidx]
      newpath = outpath / patdir / lmnadir / f'{path.stem}_{i}.png'
      torchvision.utils.save_image(torch.Tensor(this_mask), newpath)

################################################################################

class MEF_LMNA_SingleObjectMasks(MEF_LMNA_Base):
  def __init__(self, path=dfltpath_single_obj, include_cells=True, include_nuclei=True):
    # remember the dataset path
    self.root_path = Path(path)
    # retrieve and classify the samples
    self.samples = MEF_LMNA_Base.retrieve_samples(
                     path
                   , include_cells
                   , include_nuclei )

class MEF_LMNA_SingleObjectMasks_cells(MEF_LMNA_SingleObjectMasks):
  def __init__(self, path=dfltpath_single_obj):
    super().__init__(path=path, include_cells=True, include_nuclei=False)

class MEF_LMNA_SingleObjectMasks_nuclei(MEF_LMNA_SingleObjectMasks):
  def __init__(self, path=dfltpath_single_obj):
    super().__init__(path=path, include_cells=False, include_nuclei=True)

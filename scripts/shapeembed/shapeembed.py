#! /usr/bin/env python3

# machine learning utils
import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# general utils
import os
import re
import copy
import types
import pickle
import base64
import pandas
import hashlib
import logging
import datetime
import functools

# own source files
import bioimage_embed
import bioimage_embed.shapes
from dataset_transformations import *
from evaluation import *

# logging facilities
###############################################################################
logger = logging.getLogger(__name__)

# script inputs and parameters
###############################################################################

# available types of datasets (raw, masks, distance matrix)
dataset_types = [
  "raw_image"
, "mask"
, "distance_matrix"
]

# available models
models = [
  "resnet18_vae"
, "resnet50_vae"
, "resnet18_beta_vae"
, "resnet18_vae_bolt"
, "resnet50_vae_bolt"
, "resnet18_vqvae"
, "resnet50_vqvae"
, "resnet18_vqvae_legacy"
, "resnet50_vqvae_legacy"
, "resnet101_vqvae_legacy"
, "resnet110_vqvae_legacy"
, "resnet152_vqvae_legacy"
, "resnet18_vae_legacy"
, "resnet50_vae_legacy"
]

# set of parameters for a run, with default values
dflt_params = types.SimpleNamespace(
  model_name='resnet18_vae'
, dataset=types.SimpleNamespace(
    name='tiny_synthetic_shapes'
  , path='/nfs/research/uhlmann/afoix/image_datasets/tiny_synthetic_shapes'
  , type='mask'
  )
, batch_size=4
, compression_factor=2
, distance_matrix_size=512
, num_embeddings=1024
, num_hiddens=1024
, num_workers=8
, epochs=150
, pretrained=False
, frobenius_norm=False
, distance_matrix_normalize=True
, distance_matrix_roll_probability=1.0
, checkpoints_path='./checkpoints'
, commitment_cost=0.25
, decay=0.99
# optimizer_params
, opt="AdamW"
, lr=0.001
, weight_decay=0.0001
, momentum=0.9
# lr_scheduler_params
, sched="cosine"
, min_lr=1e-4
, warmup_epochs=5
, warmup_lr=1e-6
, cooldown_epochs=10
, t_max=50
, cycle_momentum=False
)

# dataset loading functions
###############################################################################

def maybe_roll(dist_mat, p = 0.5):
  if np.random.rand() < p:
    return np.roll(dist_mat, np.random.randint(0, dist_mat.shape[0]), (0,1))
  else:
    return dist_mat

def sanity_check(dist_mat):
  if not np.allclose(dist_mat, dist_mat.T):
    raise ValueError("Matrix is not symmetric")
  if np.any(dist_mat < 0):
    raise ValueError("Matrix has negative values")
  if np.any(np.diag(dist_mat)):
    raise ValueError("Matrix has non-zero diagonal")
  return dist_mat

def get_dataloader(params):

  # transformations / checks to run on distance matrices
  ts = []
  if params.distance_matrix_normalize: # optionally normalize the matrix
    ts.append(lambda x: x / np.linalg.norm(x, "fro"))
  if params.distance_matrix_roll_probability > 0.0: # optionally try to roll the matrix
    ts.append(lambda x: maybe_roll(x, p=params.distance_matrix_roll_probability))
  # always check if the matrix is symmetric, positive, and diagonal is zero
  ts.append(sanity_check)
  # turn (H,W) numpy array into a (H,W) tensor 
  ts.append(torch.as_tensor)
  # turn (H,W) tensor into a (3,H,W) tensor (downstream model expectations)
  ts.append(lambda x: x.repeat(3, 1, 1))
  # compose the all the distance matrix transformations
  logger.debug(f'transformations to run: {len(ts)}')
  distmat_ts = transforms.Compose(ts)

  # dataset to load
  logger.info(f'loading dataset {params.dataset.name}')
  dataset = None
  if params.dataset.type == 'raw_image': # TODO
    raise NotImplementedError("raw images not yet supported")
  elif params.dataset.type == 'mask': # mask data, convert to distance matrix first
    dataset = datasets.ImageFolder(
      params.dataset.path
    , transforms.Compose([ np.array
                         , functools.partial( mask2distmatrix
                                            , matrix_size=params.distance_matrix_size )
                         , distmat_ts ]))
  elif params.dataset.type == 'distance_matrix': # distance matrix data
    dataset = datasets.DatasetFolder( params.dataset.path
                                    , loader=np.load
                                    , extensions=('npy')
                                    , transform = distmat_ts )
  assert dataset, f"could not load dataset {params.dataset.name}"
  # create the dataloader from the dataset and other parameters
  dataloader = bioimage_embed.lightning.DataModule(
    dataset
  , batch_size=params.batch_size
  , shuffle=True
  , num_workers=params.num_workers
  )
  dataloader.setup()
  logger.info(f'dataloader ready')
  return dataloader

# model
###############################################################################

def get_model(params):
  logger.info(f'setup model')
  model = bioimage_embed.models.create_model(
    model=params.model_name
  , input_dim=params.input_dim
  , latent_dim=params.latent_dim
  , pretrained=params.pretrained
  , **vars(params.model_args)
  )
  lit_model = bioimage_embed.shapes.MaskEmbed(model, params)
  logger.info(f'model ready')
  return lit_model

# trainer
###############################################################################

def hashing_fn(args):
  serialized_args = pickle.dumps(vars(args))
  hash_object = hashlib.sha256(serialized_args)
  hashed_string = base64.urlsafe_b64encode(hash_object.digest()).decode()
  return hashed_string

def get_trainer(model, params):

  # setup WandB logger
  logger.info('setup wandb logger')
  jobname = f"{params.model_name}_{'_'.join([f'{k}{v}' for k, v in vars(params.model_args).items()])}_{params.latent_dim}_{params.batch_size}_{params.dataset.name}"
  wandblogger = pl_loggers.WandbLogger(entity=params.wandb_entity, project=params.wandb_project, name=jobname)
  wandblogger.watch(model, log="all")

  # setup checkpoints
  logger.info('setup checkpoints')
  model_dir = f"{params.checkpoints_path}/{hashing_fn(params)}"
  os.makedirs(f"{model_dir}/", exist_ok=True)
  checkpoint_callback = ModelCheckpoint(
    dirpath=f"{model_dir}/"
  , save_last=True
  , save_top_k=1
  , monitor="loss/val"
  , mode="min"
  )

  # setup trainer
  logger.info('setup trainer')
  trainer = pl.Trainer(
    logger=[wandblogger]
  , gradient_clip_val=0.5
  , enable_checkpointing=True
  , devices=1
  , accelerator="gpu"
  , accumulate_grad_batches=4
  , callbacks=[ checkpoint_callback
              , EarlyStopping(monitor="loss/val", mode="min")
              ]
  , min_epochs=50
  , max_epochs=params.epochs
  , log_every_n_steps=1
  )

  logger.info(f'trainer ready')
  return trainer

# train / validate / test the model
###############################################################################

def train_model(trainer, model, dataloader):
  # retrieve the checkpoint information from the trainer and check if a
  # checkpoint exists to resume from
  checkpoint_callback = trainer.checkpoint_callback
  last_checkpoint_path = checkpoint_callback.last_model_path
  best_checkpoint_path = checkpoint_callback.best_model_path
  if os.path.isfile(last_checkpoint_path):
      resume_checkpoint = last_checkpoint_path
  elif best_checkpoint_path and os.path.isfile(best_checkpoint_path):
      resume_checkpoint = best_checkpoint_path
  else:
      resume_checkpoint = None
  # train the model
  logger.info('training the model')
  trainer.fit(model, datamodule=dataloader, ckpt_path=resume_checkpoint)
  model.eval()

  return model

def validate_model(trainer, model, dataloader):
  logger.info('validating the model')
  validation = trainer.validate(model, datamodule=dataloader)
  return validation

def test_model(trainer, model, dataloader):
  logger.info('testing the model')
  testing = trainer.test(model, datamodule=dataloader)
  return testing

def run_predictions(trainer, model, dataloader, num_workers=8):

  # prepare new unshuffled datamodule
  datamod = bioimage_embed.lightning.DataModule(
    dataloader.dataset
  , batch_size=1
  , shuffle=False
  , num_workers=num_workers
  )
  datamod.setup()

  # run predictions
  predictions = trainer.predict(model, datamodule=datamod)

  # extract latent space
  latent_space = torch.stack([d.out.z.flatten() for d in predictions]).numpy()

  # extract class indices and filenames and provide a richer pandas dataframe
  ds = datamod.get_dataset()
  class_indices = np.array([ int(lbl)
                             for _, lbl in datamod.predict_dataloader() ])
  fnames = [fname for fname, _ in ds.samples]
  df = pandas.DataFrame(latent_space)
  df.insert(loc=0, column='fname', value=fnames)
  #df.insert(loc=0, column='scale', value=scalings[:,0].squeeze())
  df.insert( loc=0, column='class_name'
           , value=[ds.classes[x] for x in class_indices])
  df.insert(loc=0, column='class', value=class_indices)
  df.set_index("class", inplace=True)

  return (predictions, latent_space, df)

# main process
###############################################################################

def main_process(params):

  # setup
  #######
  model = get_model(params)
  trainer = get_trainer(model, params)
  dataloader = get_dataloader(params)

  # run actual work
  #################
  train_model(trainer, model, dataloader)
  validate_model(trainer, model, dataloader)
  test_model(trainer, model, dataloader)

  # run predictions
  #################
  # ... and gather latent space
  logger.info(f'-- run predictions and extract latent space --')
  predictions, latent_space, shapeembed_df = run_predictions(
    trainer, model, dataloader
  , num_workers=params.num_workers
  )
  logger.debug(f'\n{shapeembed_df}')
  # ... and prepare output directory and save latent space
  os.makedirs(f"{params.output_dir}/", exist_ok=True)
  np.save(f'{params.output_dir}/latent_space.npy', latent_space)
  shapeembed_df.to_pickle(f'{params.output_dir}/latent_space.pkl')
  logger.info(f'-- generate shapeembed umap --')
  umap_plot(shapeembed_df, 'shapeembed', outputdir=params.output_dir)

  # gather metrics
  ################
  # score shape embed
  logger.info(f'-- score shape embed --')
  shapeembed_score_df = score_dataframe(shapeembed_df, f'shapeembed')
  logger.info(f'-- shapeembed on input data, score:\n{shapeembed_score_df}')
  # regionprops on input data and score
  logger.info(f'-- regionprops on input data --')
  regionprops_df = run_regionprops(params.dataset)
  logger.debug(f'\n{regionprops_df}')
  regionprops_score_df = score_dataframe(regionprops_df, 'regionprops')
  logger.info(f'-- regionprops on input data, score:\n{regionprops_score_df}')
  # elliptic fourier descriptors on input data and score
  logger.info(f'-- elliptic fourier descriptors on input data --')
  efd_df = run_elliptic_fourier_descriptors(params.dataset)
  logger.debug(f'\n{efd_df}')
  efd_score_df = score_dataframe(efd_df, 'efd')
  logger.info(f'-- elliptic fourier descriptors on input data, score:\n{efd_score_df}')
  # kmeans on input data and score
  logger.info(f'-- kmeans on input data --')
  kmeans, accuracy, conf_mat = run_kmeans(dataloader_to_dataframe(dataloader.predict_dataloader()))
  print(kmeans)
  logger.info(f'-- kmeans accuracy: {accuracy}')
  logger.info(f'-- kmeans confusion matrix:\n{conf_mat}')
  # collate and save gathered results TODO KMeans
  scores_df = pandas.concat([ regionprops_score_df
                            , efd_score_df
                            , shapeembed_score_df ])
  save_scores(scores_df, outputdir=params.output_dir)

# main entry point
###############################################################################
if __name__ == '__main__':

  def auto_pos_int (x):
    val = int(x,0)
    if val <= 0:
      raise argparse.ArgumentTypeError(f"argument must be a positive int. Got {val:d}.")
    return val

  def prob (x):
    val = float(x)
    if val < 0.0 or val > 1.0:
      raise argparse.ArgumentTypeError(f"argument must be between 0.0 and 1.0. Got {val:f}.")
    return val
  
  parser = argparse.ArgumentParser(description='Run the shape embed pipeline')
  
  parser.add_argument(
      '-m', '--model', choices=models, metavar='MODEL'
    , help=f"The MODEL to use, one of {models} (default {dflt_params.model_name}).")
  parser.add_argument(
      '--model-arg-beta', type=float, metavar='BETA'
    , help=f"The BETA parameter to use for a beta-vae model.")
  parser.add_argument(
      '-d', '--dataset', nargs=3, metavar=('NAME', 'PATH', 'TYPE')
    , help=f"The NAME, PATH and TYPE of the dataset (default: {dflt_params.dataset})")
  parser.add_argument(
      '-o', '--output-dir', metavar='OUTPUT_DIR', default=None
    , help=f"The OUTPUT_DIR path to use to dump results")
  parser.add_argument(
      '--wandb-entity', default="foix", metavar='WANDB_ENTITY'
    , help=f"The WANDB_ENTITY name")
  parser.add_argument(
      '--wandb-project', default="simply-shape", metavar='WANDB_PROJECT'
    , help=f"The WANDB_PROJECT name")
  parser.add_argument(
      '-b', '--batch-size', metavar='BATCH_SIZE', type=auto_pos_int
    , help=f"The BATCH_SIZE for the run, a positive integer (default {dflt_params.batch_size})")
  parser.add_argument(
      '--distance-matrix-normalize', action=argparse.BooleanOptionalAction, default=None
    , help=f'Whether to normalize the distance matrices or not')
  parser.add_argument(
      '--distance-matrix-roll-probability', metavar='ROLL_PROB', type=prob, default=None
    , help=f'Probability to roll the distance matrices along the diagonal (default {dflt_params.distance_matrix_roll_probability})')
  parser.add_argument(
      '-c', '--compression-factor', metavar='COMPRESSION_FACTOR', type=auto_pos_int
    , help=f"The COMPRESSION_FACTOR, a positive integer (default {dflt_params.compression_factor})")
  parser.add_argument(
      '--distance-matrix-size', metavar='MATRIX_SIZE', type=auto_pos_int
    , help=f"The size of the distance matrix (default {dflt_params.distance_matrix_size})")
  parser.add_argument(
      '--number-embeddings', metavar='NUM_EMBEDDINGS', type=auto_pos_int
    , help=f"The NUM_EMBEDDINGS, a positive integer (default {dflt_params.num_embeddings})")
  parser.add_argument(
      '--number-hiddens', metavar='NUM_HIDDENS', type=auto_pos_int
    , help=f"The NUM_HIDDENS, a positive integer (default {dflt_params.num_hiddens})")
  parser.add_argument(
      '-n', '--num-workers', metavar='NUM_WORKERS', type=auto_pos_int
    , help=f"The NUM_WORKERS for the run, a positive integer (default {dflt_params.num_workers})")
  parser.add_argument(
      '-e', '--num-epochs', metavar='NUM_EPOCHS', type=auto_pos_int
    , help=f"The NUM_EPOCHS for the run, a positive integer (default {dflt_params.epochs})")
  parser.add_argument('--clear-checkpoints', action='store_true'
    , help='remove checkpoints')
  parser.add_argument('-v', '--verbose', action='count', default=0
    , help="Increase verbosity level by adding more \"v\".")

  # parse command line arguments
  clargs=parser.parse_args()

  # set verbosity level
  if clargs.verbose > 2:
    logger.setLevel(logging.DEBUG)
  elif clargs.verbose > 0:
    logger.setLevel(logging.INFO)

  # update default params with clargs
  params = copy.deepcopy(dflt_params)
  if clargs.model_name:
    params.model_name = clargs.model_name
  params.model_args = types.SimpleNamespace()
  if clargs.model_arg_beta:
    params.model_args.beta = clargs.model_arg_beta
  params.output_dir = clargs.output_dir
  if clargs.dataset:
    params.dataset = types.SimpleNamespace( name=clargs.dataset[0]
                                          , path=clargs.dataset[1]
                                          , type=clargs.dataset[2] )

  if clargs.wandb_entity:
    params.wandb_entity = clargs.wandb_entity
  if clargs.wandb_project:
    params.wandb_project = clargs.wandb_project
  if clargs.batch_size:
    params.batch_size = clargs.batch_size
  if clargs.distance_matrix_size:
    params.distance_matrix_size = clargs.distance_matrix_size
  params.input_dim = (3, params.distance_matrix_size, params.distance_matrix_size)
  if clargs.distance_matrix_normalize is not None:
    params.distance_matrix_normalize = clargs.distance_matrix_normalize
  if clargs.distance_matrix_roll_probability is not None:
    params.distance_matrix_roll_probability = clargs.distance_matrix_roll_probability
  if clargs.compression_factor:
    params.compression_factor = clargs.compression_factor
  n_features = lambda d, n: d*(d-1)/(2**n)
  params.latent_dim = n_features(params.distance_matrix_size, params.compression_factor)
  if clargs.number_embeddings:
    params.num_embeddings = clargs.number_embeddings
  if clargs.number_hiddens:
    params.num_hiddens = clargs.number_hiddens
  if clargs.num_workers:
    params.num_workers = clargs.num_workers
  if clargs.num_epochs:
    params.epochs = clargs.num_epochs
  if clargs.output_dir:
    params.output_dir = clargs.output_dir
  else:
    params.output_dir = f'./{params.model_name}_{params.latent_dim}_{params.batch_size}_{params.dataset.name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

  # XXX
  torch.set_float32_matmul_precision('medium')
  # XXX
  logger.debug(f'run parameters:\n{params}')
  main_process(params)
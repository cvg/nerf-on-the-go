# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training step and model creation functions."""

import collections
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Text, Tuple

from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image
from internal import math
from internal import models
from internal import ref_utils
from internal import robustnerf 
from internal import stepfun
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import optax

from flax import traverse_util
import jax.scipy as jsp
from functools import partial

def flattened_traversal(fn):
    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask

def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
  return tree_sum(jax.tree_util.tree_map(lambda x: jnp.sum(x**2), tree))


def tree_norm(tree):
  return jnp.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
  return jax.tree_util.tree_reduce(
      lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), tree, initializer=0)


def tree_len(tree):
  return tree_sum(
      jax.tree_util.tree_map(lambda z: jnp.prod(jnp.array(z.shape)), tree))


def summarize_tree(tree, fn, ancestry=(), max_depth=3):
  """Flatten 'tree' while 'fn'-ing values and formatting keys like/this."""
  stats = {}
  for k, v in tree.items():
    name = ancestry + (k,)
    stats['/'.join(name)] = fn(v)
    if hasattr(v, 'items') and len(ancestry) < (max_depth - 1):
      stats.update(summarize_tree(v, fn, ancestry=name, max_depth=max_depth))
  return stats

def dino_var_loss(renderings, config):
  losses = []
  for rendering in renderings:
    losses.append(jnp.mean(rendering['uncer_var']))
  return config.dino_var_mult * jnp.mean(jnp.array(losses))


from jax.lax import conv_general_dilated

def create_window(window_size, channels):
    """Create a window for SSIM computation."""
    window = jnp.ones((window_size, window_size, channels)) / (window_size**2 * channels)
    # Reshape for convolution: (spatial_dim_1, spatial_dim_2, in_channels, out_channels)
    return window.reshape(window_size, window_size, channels, 1)

def convolve(img, window):
    """Perform a convolution operation in a functional style."""
    # Define the dimension specification for the convolution operation
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    return conv_general_dilated(img, window, (1, 1), 'SAME', dimension_numbers=dimension_numbers)

def compute_ssim(img1, img2, window_size=5):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    window = create_window(window_size, 3)# size for channel

    mu1 = convolve(img1, window)
    mu2 = convolve(img2, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 * img1, window) - mu1_sq
    sigma2_sq = convolve(img2 * img2, window) - mu2_sq
    sigma12 = convolve(img1 * img2, window) - mu1_mu2

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    epsilon = jnp.finfo(jnp.float32).eps**2
    sigma1_sq = jnp.maximum(epsilon, sigma1_sq)
    sigma2_sq = jnp.maximum(epsilon, sigma2_sq)
    sigma12 = jnp.sign(sigma12) * jnp.minimum(
        jnp.sqrt(sigma1_sq * sigma2_sq), jnp.abs(sigma12))

    l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    c = (2 * jnp.sqrt(sigma1_sq) * jnp.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    s = (sigma12 + C3) / (jnp.sqrt(sigma1_sq) * jnp.sqrt(sigma2_sq) + C3)

    c = jnp.clip(c, a_max=0.98)
    s = jnp.clip(s, a_max=0.98)

    return l, c, s

def compute_data_loss(batch, renderings, rays, loss_threshold, config, train_frac):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  data_losses = []
  stats = collections.defaultdict(lambda: [])

  # lossmult can be used to apply a weight to each ray in the batch.
  # For example: masking out rays, applying the Bayer mosaic mask, upweighting
  # rays from lower resolution images and so on.
  lossmult = rays.lossmult
  lossmult = jnp.broadcast_to(lossmult, batch.rgb[..., :3].shape)
  if config.disable_multiscale_loss:
    lossmult = jnp.ones_like(lossmult)

  for rendering in renderings:
    resid_sq = (rendering['rgb'] - batch.rgb[..., :3])**2
    denom = lossmult.sum()
    stats['mses'].append((lossmult * resid_sq).sum() / denom)

    if config.data_loss_type == 'mse':
      # Mean-squared error (L2) loss.
      data_loss = resid_sq

    elif config.data_loss_type == 'on-the-go':
      uncer = rendering['uncer']
      uncer = jnp.clip(uncer, a_min=config.uncer_clip_min)+1e-3
      if config.stop_ssim_gradient:
        l,c,s = compute_ssim(jax.lax.stop_gradient(rendering['rgb']), batch.rgb[..., :3], config.ssim_window_size)
      else:
        l,c,s = compute_ssim(rendering['rgb'], batch.rgb[..., :3])
      train_frac = jnp.broadcast_to(train_frac, uncer.shape)

      # Calculate the SSIM loss rate, which starts at 100 and can scale up to 1000.
      # This is not mentioned in the paper since its effect is marginal
      bias = lambda x, s: x / (1 + (1 - x)*(1 / s - 2))
      rate = 100 + bias(train_frac, config.ssim_anneal) * 900
      my_ssim_loss = jnp.clip(rate * (1-l)*(1-s)*(1-c), a_max=config.ssim_clip_max)
      ssim_loss = my_ssim_loss / uncer**2 + config.reg_mult * jnp.log(uncer)

      # Adjust uncertainty to slowly increase based on the SSIM training fraction.
      uncer_rate = 1 + 1 * bias(train_frac, config.ssim_anneal)
      uncer = (jax.lax.stop_gradient(uncer) - config.uncer_clip_min) * uncer_rate + config.uncer_clip_min
      data_loss = 0.5 * resid_sq / (uncer) ** 2
      data_loss += config.ssim_mult * ssim_loss

    # robustnerf loss
    elif config.data_loss_type == 'robustnerf':
        mask, robust_stats = robustnerf.robustnerf_mask(resid_sq, loss_threshold, 
                config)
        data_loss = resid_sq * mask
        stats.update(robust_stats)
    else:
      assert False
    data_losses.append((lossmult * data_loss).sum() / denom)

  data_losses = jnp.array(data_losses)
  loss = (
      config.data_coarse_loss_mult * jnp.sum(data_losses[:-1]) +
      config.data_loss_mult * data_losses[-1])
  stats = {k: jnp.array(stats[k]) for k in stats}
  return loss, stats


def interlevel_loss(ray_history, config):
  """Computes the interlevel loss defined in mip-NeRF 360."""
  # Stop the gradient from the interlevel loss onto the NeRF MLP.
  last_ray_results = ray_history[-1]
  c = jax.lax.stop_gradient(last_ray_results['sdist'])
  w = jax.lax.stop_gradient(last_ray_results['weights'])
  loss_interlevel = 0.
  for ray_results in ray_history[:-1]:
    cp = ray_results['sdist']
    wp = ray_results['weights']
    loss_interlevel += jnp.mean(stepfun.lossfun_outer(c, w, cp, wp))
  return config.interlevel_loss_mult * loss_interlevel


def distortion_loss(ray_history, config):
  """Computes the distortion loss regularizer defined in mip-NeRF 360."""
  last_ray_results = ray_history[-1]
  c = last_ray_results['sdist']
  w = last_ray_results['weights']
  loss = jnp.mean(stepfun.lossfun_distortion(c, w))
  return config.distortion_loss_mult * loss


def orientation_loss(rays, model, ray_history, config):
  """Computes the orientation loss regularizer defined in ref-NeRF."""
  total_loss = 0.
  for i, ray_results in enumerate(ray_history):
    w = ray_results['weights']
    n = ray_results[config.orientation_loss_target]
    if n is None:
      raise ValueError('Normals cannot be None if orientation loss is on.')
    # Negate viewdirs to represent normalized vectors from point to camera.
    v = -1. * rays.viewdirs
    n_dot_v = (n * v[..., None, :]).sum(axis=-1)
    loss = jnp.mean((w * jnp.minimum(0.0, n_dot_v)**2).sum(axis=-1))
    if i < model.num_levels - 1:
      total_loss += config.orientation_coarse_loss_mult * loss
    else:
      total_loss += config.orientation_loss_mult * loss
  return total_loss


def predicted_normal_loss(model, ray_history, config):
  """Computes the predicted normal supervision loss defined in ref-NeRF."""
  total_loss = 0.
  for i, ray_results in enumerate(ray_history):
    w = ray_results['weights']
    n = ray_results['normals']
    n_pred = ray_results['normals_pred']
    if n is None or n_pred is None:
      raise ValueError(
          'Predicted normals and gradient normals cannot be None if '
          'predicted normal loss is on.')
    loss = jnp.mean((w * (1.0 - jnp.sum(n * n_pred, axis=-1))).sum(axis=-1))
    if i < model.num_levels - 1:
      total_loss += config.predicted_normal_coarse_loss_mult * loss
    else:
      total_loss += config.predicted_normal_loss_mult * loss
  return total_loss


def clip_gradients(grad, config):
  """Clips gradients of each MLP individually based on norm and max value."""
  # Clip the gradients of each MLP individually.
  grad_clipped = {'params': {}}
  for k, g in grad['params'].items():
    # Clip by value.
    if config.grad_max_val > 0:
      g = jax.tree_util.tree_map(
          lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), g)

    # Then clip by norm.
    if config.grad_max_norm > 0:
      mult = jnp.minimum(
          1, config.grad_max_norm / (jnp.finfo(jnp.float32).eps + tree_norm(g)))
      g = jax.tree_util.tree_map(lambda z: mult * z, g)  # pylint:disable=cell-var-from-loop

    grad_clipped['params'][k] = g
  grad = type(grad)(grad_clipped)
  return grad


def create_train_step(model: models.Model,
                      config: configs.Config,
                      dataset: Optional[datasets.Dataset] = None):
  """Creates the pmap'ed Nerf training function.

  Args:
    model: The linen model.
    config: The configuration.
    dataset: Training dataset.

  Returns:
    pmap'ed training function.
  """
  if dataset is None:
    camtype = camera_utils.ProjectionType.PERSPECTIVE
  else:
    camtype = dataset.camtype

  def train_step(
      rng,
      state,
      batch,
      cameras,
      train_frac,
      loss_threshold,
  ):
    """One optimization step.

    Args:
      rng: jnp.ndarray, random number generator.
      state: TrainState, state of the model/optimizer.
      batch: dict, a mini-batch of data for training.
      cameras: module containing camera poses.
      train_frac: float, the fraction of training that is complete.
      loss_threshold: float, the loss threshold for inliers (for robustness).

    Returns:
      A tuple (new_state, stats, rng) with
        new_state: TrainState, new training state.
        stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        rng: jnp.ndarray, updated random number generator.
    """
    rng, key, dropout_key = random.split(rng, num=3)

    def loss_fn(variables, dropout_key):
      rays = batch.rays
      if config.cast_rays_in_train_step:
        rays = camera_utils.cast_ray_batch(cameras, rays, camtype, xnp=jnp)

      # Indicates whether we need to compute output normal or depth maps in 2D.

      renderings, ray_history = model.apply(
          variables,
          key if config.randomized else None,
          rays,
          train_frac=train_frac,
          compute_extras=(),
          zero_glo=False,
          rngs={'dropout': dropout_key})
      losses = {}
      data_loss, stats = compute_data_loss(batch, renderings, rays,
              loss_threshold, config, train_frac)
      losses['data'] = data_loss

      if config.interlevel_loss_mult > 0:
        losses['interlevel'] = interlevel_loss(ray_history, config)

      if config.distortion_loss_mult > 0:
        losses['distortion'] = distortion_loss(ray_history, config)

      if (config.orientation_coarse_loss_mult > 0 or
          config.orientation_loss_mult > 0):
        losses['orientation'] = orientation_loss(rays, model, ray_history,
                                                 config)

      if (config.predicted_normal_coarse_loss_mult > 0 or
          config.predicted_normal_loss_mult > 0):
        losses['predicted_normals'] = predicted_normal_loss(
            model, ray_history, config)
      
      if config.dino_var_mult > 0:
        losses['dino_var'] = dino_var_loss(renderings, config)

      stats['weight_l2s'] = summarize_tree(variables['params'], tree_norm_sq)

      if config.weight_decay_mults:
        it = config.weight_decay_mults.items
        losses['weight'] = jnp.sum(
            jnp.array([m * stats['weight_l2s'][k] for k, m in it()]))
      stats['loss'] = jnp.sum(jnp.array(list(losses.values())))
      stats['losses'] = losses

      return stats['loss'], stats

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, stats), grad = loss_grad_fn(state.params, dropout_key)

    pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
    grad = pmean(grad)
    stats = pmean(stats)

    stats['grad_norms'] = summarize_tree(grad['params'], tree_norm)
    stats['grad_maxes'] = summarize_tree(grad['params'], tree_abs_max)

    grad = clip_gradients(grad, config)

    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)

    new_state = state.apply_gradients(grads=grad)

    opt_delta = jax.tree_util.tree_map(lambda x, y: x - y, new_state,
                                       state).params['params']
    stats['opt_update_norms'] = summarize_tree(opt_delta, tree_norm)
    stats['opt_update_maxes'] = summarize_tree(opt_delta, tree_abs_max)

    stats['psnrs'] = image.mse_to_psnr(stats['mses'])
    stats['psnr'] = stats['psnrs'][-1]
    return new_state, stats, rng

  train_pstep = jax.pmap(
      train_step,
      axis_name='batch',
      in_axes=(0, 0, 0, None, None, None),
      donate_argnums=(0, 1))
  return train_pstep


def create_optimizer(
    config: configs.Config,
    variables: FrozenVariableDict) -> Tuple[TrainState, Callable[[int], float]]:
  """Creates optax optimizer for model training."""
  adam_kwargs = {
      'b1': config.adam_beta1,
      'b2': config.adam_beta2,
      'eps': config.adam_eps,
  }
  lr_kwargs = {
      'max_steps': config.max_steps,
      'lr_delay_steps': config.lr_delay_steps,
      'lr_delay_mult': config.lr_delay_mult,
  }
 


  def get_lr_fn(lr_init, lr_final):
    return functools.partial(
        math.learning_rate_decay,
        lr_init=lr_init,
        lr_final=lr_final,
        **lr_kwargs)

  lr_fn_main = get_lr_fn(config.lr_init, config.lr_final)
  tx = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)

  return TrainState.create(apply_fn=None, params=variables, tx=tx), lr_fn_main


def create_render_fn(model: models.Model):
  """Creates pmap'ed function for full image rendering."""

  def render_eval_fn(variables, train_frac, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            train_frac=train_frac,
            compute_extras=True,
            is_training=False),
        axis_name='batch')

  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, None, 0),
      axis_name='batch',
  )
  return render_eval_pfn


def setup_model(
    config: configs.Config,
    rng: jnp.array,
    dataset: Optional[datasets.Dataset] = None,
) -> Tuple[models.Model, TrainState, Callable[
    [FrozenVariableDict, jnp.array, utils.Rays],
    MutableMapping[Text, Any]], Callable[
        [jnp.array, TrainState, utils.Batch,
            Optional[Tuple[Any, ...]], float, float],
        Tuple[TrainState, Dict[Text, Any], jnp.array]], Callable[[int], float]]:
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""
  feat_dim = config.feat_dim
  dummy_rays = utils.dummy_rays(
    feat_dim, include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
  model, variables = models.construct_model(rng, dummy_rays, config)

  state, lr_fn = create_optimizer(config, variables)
  render_eval_pfn = create_render_fn(model)
  train_pstep = create_train_step(model, config, dataset=dataset)

  return model, state, render_eval_pfn, train_pstep, lr_fn

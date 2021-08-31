#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


def plot_pr_curves(pr_curves, dataset_name, save_dir):
  '''
  plot pr curves
  input:
    pr_curves: Dict[interval: pr_curve]
    dataset_name: "dataset" + "seq"
    save_dir: save directory
  '''
  
  plt.title(dataset_name)
  colors = ['green', 'red', 'blue', 'yellow', 'darkviolet', 'sandybrown']
  for k, c in zip(pr_curves.keys(), colors):
    pr_curve = pr_curves[k]
    xs, ys = [], []
    for pr in pr_curve:
      xs.append(pr[0])
      ys.append(pr[1])

    plt.plot(xs, ys, color=c, label=str(k))

  plt.legend()
  plt.xlabel('precision')
  plt.ylabel('recall')

  image_name = dataset_name + ".jpg"
  save_path = os.path.join(save_dir, image_name)
  plt.savefig(save_path)


def get_pr_curve_area(pr_curve):
  '''
  pr_curve: [[p0, r0], [p1, r1]... [pn, rn]], thr: small->big, precision: small->big, recall: big->small
  '''
  area = 0.0
  for i in range(1, len(pr_curve)):
    p0, r0 = pr_curve[i-1]
    p1, r1 = pr_curve[i]

    area = area + (r0 - r1) * (p1 + p0) / 2

  return area    


def plot_tracking_details(results_list, save_dir, name=None, configs=None):
  if config is not None:
    title = configs['title']
    colors = configs['colors']
    linewidth = configs['linewidth']
    xlabel = configs['xlabel']
    ylabel = configs['ylabel']
    fontsize = configs['fontsize']
    figsize = configs['figsize']
    dpi = configs['dpi']
  else:
    title = results_list[0]['dataset'] if name is None else name
    colors = ['green', 'red', 'blue', 'yellow', 'darkviolet', 'sandybrown']
    linewidth = 3
    xlabel = "recall"
    ylabel = "precision"
    fontsize = 20
    figsize = (10, 10)
    dpi = 100

  plt.title(title)
  plt.xticks(fontsize=fontsize)
  plt.yticks(fontsize=fontsize)

  for i in range(len(results_list)):
    pr_curves = results_list[i]['pr_curves']
    areas = results_list[i]['areas']
    for k in pr_curves.keys():
      pr_curve = pr_curves[k]
      xs, ys = [], []
      for pr in pr_curve:
        xs.append(pr[1]) # recall
        ys.append(pr[0]) # precision
      
      area = round(areas[k], 4)

      label = "[{}] k = {}, {}".format(area, k, results_list[i]['model'])
      linestyle = '-' if i==0 else '--'
      plt.plot(xs, ys, color=colors[k], label=label, linewidth=linewidth, linestyle=linestyle)

  plt.legend(fontsize=fontsize)
  plt.grid()
  plt.xlabel(xlabel, fontsize=fontsize)
  plt.ylabel(ylabel, fontsize=fontsize)
  
  image_name = title + ".jpg"
  save_path = os.path.join(save_dir, image_name)
  plt.savefig(save_path)


def save_tracking_results(results, save_dir):
  '''
  saving tracking experiment results

  results:
    dataset: *
    model: *
    pr_curves:
      interval_0: [[p00, r00], [p01, r01]... [p0n, r0n]]
      interval_1: [[p10, r10], [p11, r11]... [p1n, r1n]]
      ...
      interval_m: [[pm0, rm0], [pm1, rm1]... [pmn, rmn]]
  '''
  file_name = results['dataset'] + "_" + results['model'] + ".yaml"
  file_path = os.path.join(save_dir, file_name)
  fp = open(file_path, 'w')
  fp.write(yaml.dump(results))


def read_tracking_results(file_path):
  f = open(config_file, 'r', encoding='utf-8')
  results = f.read()
  f.close()
  return results
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import os
import yaml
import numpy as np
import argparse

from experiments.utils.utils import read_tracking_results, plot_tracking_details


def filter_pr_curves(results, plot_keys):
  pr_curves = results['pr_curves']
  new_pr_curves = {}
  for k in pr_curves.keys():
    if k in plot_keys:
      new_pr_curves[k] = pr_curves[k]

  results['pr_curves'] = new_pr_curves
  return results


def compare_tracking(configs):
  f1 = configs['file1']
  f2 = configs['file2']
  save_dir = configs['save_dir']
  plot_keys = configs['interval']

  results1 = read_tracking_results(f1)
  results2 = read_tracking_results(f2)
  
  results1 = filter_pr_curves(results1, plot_keys)
  results2 = filter_pr_curves(results2, plot_keys)

  results_list = [results1, results2]
  plot_tracking_details(results_list, save_dir, configs=configs)


def main():
  parser = argparse.ArgumentParser(description="compare tracking results")
  parser.add_argument(
      "-f1", "--file1",
      dest = "file1",
      type = str, 
      default = ""
  )
  parser.add_argument(
      "-f2", "--file2",
      dest = "file2",
      type = str, 
      default = ""
  )
  parser.add_argument(
      "-s", "--save_dir",
      dest = "save_dir",
      type = str, 
      default = "" 
  )
  parser.add_argument(
      "-c", "--config_file",
      dest = "config_file",
      type = str, 
      default = ""
  )
  args = parser.parse_args()
  config_file = args.config_file
  f = open(config_file, 'r', encoding='utf-8')
  configs = f.read()
  configs = yaml.load(configs)
  configs['file1'] = args.file1
  configs['file2'] = args.file2
  configs['save_dir'] = args.save_dir

  compare_tracking(configs)

if __name__ == "__main__":
  main()
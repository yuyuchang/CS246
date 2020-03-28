# -*- coding: utf-8 -*-
"""q4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12_mPHXxBXmqGjk4PAYeEN_oj50-sJKEn
"""

import numpy as np
from numpy.linalg import inv

R = np.loadtxt('user-shows.txt')
P = np.zeros((R.shape[0], R.shape[0]))
Q = np.zeros((R.shape[1], R.shape[1]))

shows_file = 'shows.txt'
shows_list = []

f = open(shows_file, 'r')
for line in f:
  shows_name = line.strip().replace('"', '')
  shows_list.append(shows_name)

R_row_sum = R.sum(axis = 0)
R_col_sum = R.sum(axis = 1)

for i in range(len(R_col_sum)):
  P[i, i] = R_col_sum[i]

for i in range(len(R_row_sum)):
  Q[i, i] = R_row_sum[i]

Gamma_uu = (inv(P) ** 0.5) @ R @ R.T @(inv(P) ** 0.5) @ R
Gamma_ii = R @(inv(Q) ** 0.5) @ R.T @ R @ (inv(Q) ** 0.5)

top5_uu_indexes = (-Gamma_uu[499, :100]).argsort()[:5]
print("The names of five TV shows that have the highest similarity scores for Alex \nfor the user-user collaborative filtering are:\n")
for index in top5_uu_indexes:
  print("{}, with similarity score {}".format(shows_list[index], Gamma_uu[499, index]))

print("=" * 30)

top5_ii_indexes = (-Gamma_ii[499, :100]).argsort()[:5]
print("The names of five TV shows that have the highest similarity scores for Alex \nfor the item-item collaborative filtering are:\n")
for index in top5_ii_indexes:
  print("{}, with similarity score {}".format(shows_list[index], Gamma_ii[499, index]))


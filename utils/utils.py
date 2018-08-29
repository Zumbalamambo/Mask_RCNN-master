# -*- coding: utf-8 -*-
"""
Created on 2018/8/7 9:33
@author: gzp
Func: 
"""
import os, sys, math
sys.path.append('../')  # add root path of the project

def get_files(file_dir):
    return [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))]
def get_dirs(file_dir):
    return [file for file in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, file))]
def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()
def make_dir(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
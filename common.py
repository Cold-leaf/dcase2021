########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import itertools
import re

# additional
import numpy as np
import librosa  # 音频处理
import librosa.core
import librosa.feature
import yaml  # yaml 专门用来写配置文件

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")       #输出日志等级、输出文件名
logger = logging.getLogger(' ')
handler = logging.StreamHandler()               #流handler  能够将日志信息输出到sys.stdout, sys.stderr 或者类文件对象
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')      #日期格式
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################


########################################################################
# argparse      命令行选项、参数和子命令解析器
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')      #创建解析器 description是在参数帮助文档之前显示的文本
    parser.add_argument('-v', '--version', action='store_true', help="show application version")            #添加参数(action - 当参数在命令行中出现时使用的动作基本类型; help - 一个此选项作用的简单描述)
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    args = parser.parse_args()              #解析参数 
    if args.version:
        print("===============================")
        print("DCASE 2021 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.dev:
        flag = True
    elif args.eval:
        flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml       加载baseline.yaml（包含各参数）
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file input    加载音频（wav）为numpy，返回numpy和sr（采样率）
def file_load(wav_name, mono=False):
    """
    加载 .wav 音频文件

    wav_name : str
        目标音频文件
    mono : boolean
        mono 为真时, 多声道的音频会被合并成单声道
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)           #path、采样率、通道（false：双通道）
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0):
    """
    将 file_name 转换为 vector 数组

    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # 使用 librosa 生成 mel 频谱
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,           #mel频谱
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energies
    # 将 mel 频谱转化为对数标度
    # sys.float_info.epsilon 为 python 浮点数的最小差值
    # np.maximun 在此将 mel_spectrogram 中小于 epsilon 的值截断
    # 处理得到的数组逐元素取对数, 乘以 20 / power 就得到对数标度下的 mel 频谱
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

    return vectors


########################################################################


########################################################################
# get directory paths according to mode
########################################################################
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


########################################################################


########################################################################
# get machine IDs
########################################################################
def get_section_names(target_dir,
                      dir_name,
                      ext="wav"):
    """
    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files

    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return section_names


########################################################################


########################################################################
# get the list of wave file paths
########################################################################
def file_list_generator(target_dir,
                        section_name,
                        dir_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))

    # development
    if mode:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_normal,
                                                                                                     ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_anomaly,
                                                                                                     ext=ext))
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     ext=ext))
        files = sorted(glob.glob(query))
        labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################

import os
import os.path as osp
import random
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import glob_file_list

def sep_SMID(data_path, save_path):
    """Separate SMID dataset.

        Args:
            data_path (str): Path to SMID.
            save_path (str): Path to save dataset.
    """
    if not os.path.isdir(data_path):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    test_list = ['0013', '0009', '0020', '0007', '0010', '0012', '0154', '0036', '0037', '0039', '0047', '0048', '0049',
                 '0050', '0051', '0052', '0065', '0066', '0075', '0076', '0078', '0088', '0092', '0099', '0091', '0103',
                 '0104', '0105', '0106', '0107', '0151', '0145', '0147', '0139', '0129', '0083', '0153', '0157', '0180',
                 '0175', '0181', '0196', '0170', '0166', '0172', '0177', '0167', '0169', '0191']

    if not osp.exists(save_path + '/train'):
        os.mkdir(save_path + '/train')
        os.mkdir(save_path + '/train/LQ')
        os.mkdir(save_path + '/train/Long')
    if not osp.exists(save_path + '/test'):
        os.mkdir(save_path + '/test')
        os.mkdir(save_path + '/test/LQ')
        os.mkdir(save_path + '/test/Long')

    data_LQ_path = data_path + '/SMID_LQ_np'
    data_GT_path = data_path + '/SMID_Long_np'

    data_LQ_pairs_path = glob_file_list(data_LQ_path)
    data_GT_pairs_path = glob_file_list(data_GT_path)

    videos_LQ = []
    videos_GT = []
    videos_test_LQ = {}
    videos_test_GT = {}

    for pairX_LQ, pairX_GT in zip(data_LQ_pairs_path, data_GT_pairs_path):
        pair_name = osp.basename(pairX_LQ)
        if pair_name in test_list:
            video_LQ = glob_file_list(pairX_LQ)
            video_GT = glob_file_list(pairX_GT)
            video_LQ = video_LQ[0:30]
            frames_number = len(video_LQ)
            if frames_number % 7 == 0:
                count = 0
                for shortvideo_start_index in range(0, frames_number, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    videos_test_GT[pair_name + '_' + str(count)] = video_GT
            else:
                frames_number_7 = int(frames_number / 7)
                lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
                count = 0
                for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    videos_test_GT[pair_name + '_' + str(count)] = video_GT

                shortvideo_LQ = []
                count += 1
                for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                videos_test_GT[pair_name + '_' + str(count)] = video_GT

            continue

        video_LQ = glob_file_list(pairX_LQ)
        video_GT = glob_file_list(pairX_GT)
        frames_number = len(video_LQ)
        if frames_number % 7 == 0:
            for shortvideo_start_index in range(0, frames_number, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                videos_LQ.append(shortvideo_LQ)
                videos_GT.append(video_GT)
        else:
            frames_number_7 = int(frames_number / 7)
            lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
            for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                videos_LQ.append(shortvideo_LQ)
                videos_GT.append(video_GT)

            shortvideo_LQ = []
            for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
            videos_LQ.append(shortvideo_LQ)
            videos_GT.append(video_GT)

    videos_number = len(videos_GT)
    for pair_index in range(1, len(videos_GT) + 1):
        save_path_LQ_pairX = save_path + '/train/LQ/pair' + str(pair_index)
        save_path_GT_pairX = save_path + '/train/Long/pair' + str(pair_index)

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)
        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        choose_video_index = random.randint(0, videos_number-1)
        choose_shortvideo_LQ = videos_LQ.pop(choose_video_index)
        choose_shortvideo_GT = videos_GT.pop(choose_video_index)
        videos_number -= 1

        count = 1
        for src_dir_LQ in choose_shortvideo_LQ:
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            shutil.copyfile(src_dir_LQ, dst_dir_LQ)
        dst_dir_GT = osp.join(save_path_GT_pairX, '01.npy')
        shutil.copyfile(choose_shortvideo_GT[0], dst_dir_GT)

    for k, v in videos_test_LQ.items():
        save_path_LQ_pairX = save_path + '/test/LQ/' + k

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)

        count = 1
        for src_dir_LQ in v:
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            shutil.copyfile(src_dir_LQ, dst_dir_LQ)

    for k, v in videos_test_GT.items():
        save_path_GT_pairX = save_path + '/test/Long/' + k

        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        count = 1
        for src_dir_GT in v:
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_GT = osp.join(save_path_GT_pairX, new_filename)
            shutil.copyfile(src_dir_GT, dst_dir_GT)


def main():
    sep_SMID('/path_to_smid_np', '/path_to_dataset')

if __name__ == "__main__":
    main()
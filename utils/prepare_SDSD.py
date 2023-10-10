import os
import os.path as osp
import random
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import glob_file_list

def sep_SDSD(indoor_data_path, outdoor_data_path, save_path):
    """Separate SDSD dataset.

        Args:
            indoor_data_path (str): Path to SDSD indoor part.
            outdoor_data_path (str): Path to SDSD outdoor part.
            save_path (str): Path to save dataset.
    """
    if not os.path.isdir(indoor_data_path):
        print('Error: No source indoor_data found')
        exit(0)
    elif not os.path.isdir(outdoor_data_path):
        print('Error: No source outdoor_data found')
        exit(0)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    indoor_test = ['pair13', 'pair15', 'pair21', 'pair23', 'pair31', 'pair33', 'pair50', 'pair52', 'pair58', 'pair60', 'pair68', 'pair70']
    outdoor_test = ['pair1', 'pair5', 'pair14', 'pair36', 'pair46', 'pair48', 'pair49', 'pair60', 'pair62', 'pair63', 'pair66', 'pair75', 'pair76']

    if not osp.exists(save_path + '/indoor_np'):
        os.mkdir(save_path + '/indoor_np')
        os.mkdir(save_path + '/indoor_np/train')
        os.mkdir(save_path + '/indoor_np/train/input')
        os.mkdir(save_path + '/indoor_np/train/GT')
        os.mkdir(save_path + '/indoor_np/test')
        os.mkdir(save_path + '/indoor_np/test/input')
        os.mkdir(save_path + '/indoor_np/test/GT')
    if not osp.exists(save_path + '/outdoor_np'):
        os.mkdir(save_path + '/outdoor_np')
        os.mkdir(save_path + '/outdoor_np/train')
        os.mkdir(save_path + '/outdoor_np/train/input')
        os.mkdir(save_path + '/outdoor_np/train/GT')
        os.mkdir(save_path + '/outdoor_np/test')
        os.mkdir(save_path + '/outdoor_np/test/input')
        os.mkdir(save_path + '/outdoor_np/test/GT')

    #indoor_np/input, indoor_np/GT, outdoor_np/input, outdoor_np/GT
    indoor_data_LQ_path = indoor_data_path + '/input'
    indoor_data_GT_path = indoor_data_path + '/GT'
    outdoor_data_LQ_path = outdoor_data_path + '/input'
    outdoor_data_GT_path = outdoor_data_path + '/GT'

    #indoor_np/input/pairX
    indoor_data_LQ_pairs_path = glob_file_list(indoor_data_LQ_path)
    indoor_data_GT_pairs_path = glob_file_list(indoor_data_GT_path)
    outdoor_data_LQ_pairs_path = glob_file_list(outdoor_data_LQ_path)
    outdoor_data_GT_pairs_path = glob_file_list(outdoor_data_GT_path)

    indoor_videos_LQ = []
    indoor_videos_GT = []
    indoor_videos_test_LQ = {}
    indoor_videos_test_GT = {}
    outdoor_videos_LQ = []
    outdoor_videos_GT = []
    outdoor_videos_test_LQ = {}
    outdoor_videos_test_GT = {}

    for indoor_pairX_LQ, indoor_pairX_GT in zip(indoor_data_LQ_pairs_path, indoor_data_GT_pairs_path):
        pair_name = osp.basename(indoor_pairX_GT)
        if pair_name in indoor_test:
            video_LQ = glob_file_list(indoor_pairX_LQ)
            video_GT = glob_file_list(indoor_pairX_GT)
            video_LQ = video_LQ[0:30]
            video_GT = video_GT[0:30]
            frames_number = len(video_GT)
            if frames_number % 7 == 0:
                count = 0
                for shortvideo_start_index in range(0, frames_number, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    shortvideo_GT = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                        shortvideo_GT.append(video_GT[shortvideo_frame_index])
                    indoor_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    indoor_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT
            else:
                frames_number_7 = int(frames_number / 7)
                lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
                count = 0
                for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    shortvideo_GT = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                        shortvideo_GT.append(video_GT[shortvideo_frame_index])
                    indoor_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    indoor_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT

                shortvideo_LQ = []
                shortvideo_GT = []
                count += 1
                for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                indoor_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                indoor_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT

            continue

        video_LQ = glob_file_list(indoor_pairX_LQ)
        video_GT = glob_file_list(indoor_pairX_GT)
        frames_number = len(video_GT)
        if frames_number % 7 == 0:
            for shortvideo_start_index in range(0, frames_number, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                shortvideo_GT = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                indoor_videos_LQ.append(shortvideo_LQ)
                indoor_videos_GT.append(shortvideo_GT)
        else:
            frames_number_7 = int(frames_number / 7)
            lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
            for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                shortvideo_GT = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                indoor_videos_LQ.append(shortvideo_LQ)
                indoor_videos_GT.append(shortvideo_GT)

            shortvideo_LQ = []
            shortvideo_GT = []
            for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                shortvideo_GT.append(video_GT[shortvideo_frame_index])
            indoor_videos_LQ.append(shortvideo_LQ)
            indoor_videos_GT.append(shortvideo_GT)

    for outdoor_pairX_LQ, outdoor_pairX_GT in zip(outdoor_data_LQ_pairs_path, outdoor_data_GT_pairs_path):
        pair_name = osp.basename(outdoor_pairX_GT)
        if pair_name in outdoor_test:
            video_LQ = glob_file_list(outdoor_pairX_LQ)
            video_GT = glob_file_list(outdoor_pairX_GT)
            video_LQ = video_LQ[0:30]
            video_GT = video_GT[0:30]
            frames_number = len(video_GT)
            if frames_number % 7 == 0:
                count = 0
                for shortvideo_start_index in range(0, frames_number, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    shortvideo_GT = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                        shortvideo_GT.append(video_GT[shortvideo_frame_index])
                    outdoor_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    outdoor_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT
            else:
                frames_number_7 = int(frames_number / 7)
                lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
                count = 0
                for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    shortvideo_GT = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                        shortvideo_GT.append(video_GT[shortvideo_frame_index])
                    outdoor_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    outdoor_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT

                shortvideo_LQ = []
                shortvideo_GT = []
                count += 1
                for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                outdoor_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                outdoor_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT

            continue

        video_LQ = glob_file_list(outdoor_pairX_LQ)
        video_GT = glob_file_list(outdoor_pairX_GT)
        frames_number = len(video_GT)
        if frames_number % 7 == 0:
            for shortvideo_start_index in range(0, frames_number, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                shortvideo_GT = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                outdoor_videos_LQ.append(shortvideo_LQ)
                outdoor_videos_GT.append(shortvideo_GT)
        else:
            frames_number_7 = int(frames_number / 7)
            lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
            for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                shortvideo_GT = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                outdoor_videos_LQ.append(shortvideo_LQ)
                outdoor_videos_GT.append(shortvideo_GT)

            shortvideo_LQ = []
            shortvideo_GT = []
            for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                shortvideo_GT.append(video_GT[shortvideo_frame_index])
            outdoor_videos_LQ.append(shortvideo_LQ)
            outdoor_videos_GT.append(shortvideo_GT)

    videos_number = len(indoor_videos_GT)
    for pair_index in range(1, len(indoor_videos_GT) + 1):
        save_path_LQ_pairX = save_path + '/indoor_np/train/input/pair' + str(pair_index)
        save_path_GT_pairX = save_path + '/indoor_np/train/GT/pair' + str(pair_index)

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)
        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        choose_video_index = random.randint(0, videos_number-1)
        choose_shortvideo_LQ = indoor_videos_LQ.pop(choose_video_index)
        choose_shortvideo_GT = indoor_videos_GT.pop(choose_video_index)
        videos_number -= 1

        count = 1
        for src_dir_LQ, src_dir_GT in zip(choose_shortvideo_LQ, choose_shortvideo_GT):
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            dst_dir_GT = osp.join(save_path_GT_pairX, new_filename)

            shutil.copyfile(src_dir_LQ, dst_dir_LQ)
            shutil.copyfile(src_dir_GT, dst_dir_GT)

    for k, v in indoor_videos_test_LQ.items():
        save_path_LQ_pairX = save_path + '/indoor_np/test/input/' + k

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)

        count = 1
        for src_dir_LQ in v:
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            shutil.copyfile(src_dir_LQ, dst_dir_LQ)

    for k, v in indoor_videos_test_GT.items():
        save_path_GT_pairX = save_path + '/indoor_np/test/GT/' + k

        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        count = 1
        for src_dir_GT in v:
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_GT = osp.join(save_path_GT_pairX, new_filename)
            shutil.copyfile(src_dir_GT, dst_dir_GT)

    videos_number = len(outdoor_videos_GT)
    for pair_index in range(1, len(outdoor_videos_GT) + 1):
        save_path_LQ_pairX = save_path + '/outdoor_np/train/input/pair' + str(pair_index)
        save_path_GT_pairX = save_path + '/outdoor_np/train/GT/pair' + str(pair_index)

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)
        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        choose_video_index = random.randint(0, videos_number - 1)
        choose_shortvideo_LQ = outdoor_videos_LQ.pop(choose_video_index)
        choose_shortvideo_GT = outdoor_videos_GT.pop(choose_video_index)
        videos_number -= 1

        count = 1
        for src_dir_LQ, src_dir_GT in zip(choose_shortvideo_LQ, choose_shortvideo_GT):
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            dst_dir_GT = osp.join(save_path_GT_pairX, new_filename)

            shutil.copyfile(src_dir_LQ, dst_dir_LQ)
            shutil.copyfile(src_dir_GT, dst_dir_GT)

    for k, v in outdoor_videos_test_LQ.items():
        save_path_LQ_pairX = save_path + '/outdoor_np/test/input/' + k

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)

        count = 1
        for src_dir_LQ in v:
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            shutil.copyfile(src_dir_LQ, dst_dir_LQ)

    for k, v in outdoor_videos_test_GT.items():
        save_path_GT_pairX = save_path + '/outdoor_np/test/GT/' + k

        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        count = 1
        for src_dir_GT in v:
            new_filename = '0' + str(count) + '.npy'
            count += 1

            dst_dir_GT = osp.join(save_path_GT_pairX, new_filename)
            shutil.copyfile(src_dir_GT, dst_dir_GT)


def main():
    sep_SDSD('/path_to_indoor_np', '/path_to_outdoor_np', '/path_to_dataset')

if __name__ == "__main__":
    main()
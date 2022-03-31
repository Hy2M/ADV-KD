import os
import shutil
import sys

path_root = './results/'
for _, lstFldr, _ in os.walk(path_root):
    break

for i in range(len(lstFldr)):
    path_log = path_root + lstFldr[i] + '/log/log.txt'
    print('-----------------------')
    print(path_log)
    with open(path_log, 'r') as f:
        
        lines_cnt = 0
        max_top_1 = 0.
        max_top_5 = 0.
        min_top_ecc = 0.

        lines = f.readlines()
        for i in range(len(lines)):
            line = str(lines[i])
            if lines_cnt < 1:
                print(line)
            lines_cnt += 1

            line_st_t1 = line.find('[val_top1_acc')
            line_st_t5 = line.find('] [val_top5_acc')
            line_st_ece = line.find('] [ECE')
            line_ed_ece = line.find('] [AURC')

            if line_st_t1 > 0 or line_st_t5 > 0:
                top_1 = float(line[line_st_t1+14:line_st_t5])
                top_5 = float(line[line_st_t5+16:line_st_ece])
                ece = float(line[line_st_ece+7:line_ed_ece])

                if top_1 > max_top_1:
                    max_top_1 = top_1
                    max_top_5 = top_5
                    min_ece = ece

        print(max_top_1, max_top_5, min_ece)

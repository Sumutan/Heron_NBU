# import csv
#
# inputfile='cloudval.csv'
# ontputfile='valsort.csv'
#
# with open(inputfile, 'r') as input_file, open(ontputfile, 'w', newline='') as output_file:
#     reader = csv.reader(input_file)
#     writer = csv.writer(output_file)
#     for row in reader:
#         sorted_row = sorted(row)
#         writer.writerow(sorted_row)


import csv

inputfile='val.csv'
ontputfile='valsort_nochange.csv'
cloud_head='/home/ma-user/work/dataset/ucf101/val/val/'
zhang_head='/home/zhangsenzhen/DATASET/UCF101/UCF-101/'


with open(inputfile, 'r') as input_file, open(ontputfile, 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    sorted_rows = []
    for row in reader:
        # 删除开头的路径
        row = [x.replace(zhang_head, cloud_head) for x in row]
        sorted_rows.append(sorted(row))
    # 对所有行按字符升序进行排序
    sorted_rows = sorted(sorted_rows)
    for sorted_row in sorted_rows:
        writer.writerow(sorted_row)
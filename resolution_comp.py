import csv
import os

def data_comp(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        header = reader[0]
        data = reader[1:]

        # 5의 배수 (인덱스 기준 1부터 시작한다고 가정)
        filtered_data = [row for idx, row in enumerate(data, start=1) if idx % 5 == 1]

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(filtered_data)

    print(f"{output_file} 파일에 저장 완료!")

dir_name = './dataset/data_5us'
for fname in os.listdir(dir_name):
    if fname.endswith('.csv'):
        dir_fname = dir_name+'/'+fname
        data_comp(dir_fname, dir_fname)
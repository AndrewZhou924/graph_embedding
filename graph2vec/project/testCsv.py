import csv

# ---------------------------------------------------------------
# #csv 写入
# stu1 = [1,2,4]
# #打开文件，追加a
# out = open('Stu_csv.csv','a', newline='')


# #设定写入模式
# # csv_write = csv.writer(out,dialect='excel')

# csv_write = csv.writer(out)
# # csv_write.writerow(["vec","label"])

# #写入具体内容
# csv_write.writerow(stu1)
# # csv_write.writerow(stu2)
# print ("write over")


# ---------------------------------------------------------------
# read csv
# reader = csv.reader(open("./Stu_csv.csv")
# print(reader)

import pandas as pd
data = pd.read_csv("./Stu_csv.csv")
# print(data['vec'].shape)
# print(data['label'])


# print(data)
# for index, row in data.iterrows():
#     print(row['x1'])

# 查找 & 判断
print ((len(data[data.index == 10])) == 0)
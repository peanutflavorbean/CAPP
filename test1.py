import csv
file = open('/Users/imac/anaconda3/test1/Rrod_tolerance.csv','r')
rdr = csv.reader(file)
for line in rdr:
    print(line)

file.close()
import xlrd

workbook = xlrd.open_workbook('spam_ham_dataset.xlsx')
worksheet = workbook.sheet_by_name('spam_ham_dataset')
num_rows = worksheet.nrows - 1
num_colums = worksheet.ncols - 1
content = ''
print(num_colums)
print(num_rows)
column = 2
curr_row = 2
while curr_row < num_rows:
        curr_row += 1
        content += worksheet.cell(curr_row, column).value
contentArray = content.split(" ")
nonDuplicateContent = list(set(contentArray))

# print(contentArray)
frequentWords = {}
for index, word in enumerate(nonDuplicateContent):
        frequentWords[word] = contentArray.count(word)

sorted = {k: v for k, v in sorted(frequentWords.items(), key=lambda item: item[1], reverse = True)}
print(sorted)

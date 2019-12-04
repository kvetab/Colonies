import csv
import matplotlib.pyplot as plt
import re


filters = []
function_sum = []
split = []
sums = []
yvalues = []
yvalues_abs = []
best_filters, best_split, best_func = '', '', ''
err = 8
# dodat cestu
with open('results_.csv', 'r') as _filehandler:
    reader = csv.DictReader(_filehandler)
    for row in reader:
        error = float(row['error'])
        if len(row['abs_error']) > 0:
            abs_error = float(row['abs_error'])
        else:
            abs_error = None
        if int(row['epochs']) >= 100:
            function_sum.append(row['outf_s'])
            yvalues.append(error)
            yvalues_abs.append(abs_error)
            filter_nums = [int(s) for s in re.findall(r'\d+', row['filters'])]
            filters.append(filter_nums)
            split.append(row['split'])
            sums.append(row['sums'])
            if error < err:
                err = error
                best_filters, best_func, best_split, best_sum = row['filters'], row['outf_s'], row['split'], row['sums']


def GetUniqueVals(raw):
    output = []
    for x in raw:
        if x not in output:
            output.append(x)
    output.sort()
    dict_err = {}
    dict_abs = {}
    for uniq in output:
        dict_err.update({str(uniq): []})
        dict_abs.update({str(uniq): []})
    for i, piece in enumerate(raw):
        dict_err[str(piece)].append(yvalues[i])
        dict_abs[str(piece)].append(yvalues_abs[i])
    return dict_err, dict_abs


def AvgList(llist):
    res = []
    for elem in llist:
        filtered = list(filter(None, elem))
        if len(filtered) > 0:
            res.append(sum(filtered)/len(filtered))
        else:
            res.append(0)
    return res


def show_chart(x_labels, x_lab_abs, name):
    x_coords = list(range(1, len(x_labels)+1))
    x_ticks = x_labels.keys()
    y1 = AvgList(x_labels.values())
    y2 = AvgList(x_lab_abs.values())

    plt.xticks(x_coords, x_ticks)

    plt.plot(x_coords, y1, label="error")
    plt.plot(x_coords, y2, label="abs_error")
    plt.legend()

    plt.xlabel(name)
    plt.ylabel('error value')
    plt.show()


#print("Mean error for all filters passing values: % 9.8f, absolute error: % 9.8f" % (splitF['error'], splitF['abs_error']))
#print("Mean error for split filters: %9.8f, absolute error: %9.8f" % (splitT['error'], splitT['abs_error']))
err, abs = GetUniqueVals(split)
show_chart(err, abs, "split filters")
err, abs = GetUniqueVals(filters)
show_chart(err, abs, "Filter numbers")
err, abs = GetUniqueVals(function_sum)
show_chart(err, abs, "Output function")
err, abs = GetUniqueVals(sums)
show_chart(err, abs, "what to sum")

print(best_filters, best_func, best_split, best_sum)




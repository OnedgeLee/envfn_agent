import tensorflow as tf

def progress_bar(current, total, prefix='', suffix='', decimals=1, length=20, bar=u"\u25AF", fill=u"\u25AE"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filledLength = int(length * current // total)
    bar = fill * filledLength + bar * (length - filledLength)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if current == total: 
        print()

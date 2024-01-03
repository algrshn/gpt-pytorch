import sys


def display_progress(batch_num, num_of_batches):
    
    total=num_of_batches-1
    bar_len = 60
    filled_len = int(round(bar_len * batch_num / float(total)))

    percents = round(100.0 * batch_num / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()
    
    if(percents == 100.0):
        sys.stdout.write('')
        sys.stdout.flush()

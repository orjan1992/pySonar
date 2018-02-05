def print_args(**kwargs):
    tmp = ''
    for a in kwargs:
        tmp = '{}{}: {}\t'.format(tmp, a, kwargs[a])
    print(tmp)

import sys

for line in sys.stdin:
    arr = line.split('\t')
    if len(arr) == 2:
        print('\n'.join(arr))

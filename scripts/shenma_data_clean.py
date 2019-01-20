import sys
for line in sys.stdin:
    arr = line.strip().split('\t')
    if len(arr) == 2 and 10 <= len(arr[0]) <= 20 and 10 <= len(arr[1]) <= 20:
        sys.stdout.write(line)

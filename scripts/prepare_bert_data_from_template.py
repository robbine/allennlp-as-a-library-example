import sys

def process_line(line):
    arr = line.split(' ')
    return arr[0], ''.join(arr[1:])

prev_label = ''
for line in sys.stdin:
    label, content = process_line(line.strip())
    if label != prev_label:
        print('')
        prev_label = label
    print(content)

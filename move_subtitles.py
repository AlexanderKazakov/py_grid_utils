import re

# на сколько сдвинуть субтитры относительно фильма вперёд в миллисекундах
time_to_move_forward_ms = -12500
filename = r"C:\Users\alex\Downloads\from dusk.srt"


def parse_time_ms(time):
    time = ':' + time
    h, m, s = tuple([int(val) for val in re.findall(r':(\d{2})', time)])
    ms = float(re.search(r',\d+', time).group(0).replace(',', '.'))
    ms = int(1000 * ms)
    return 1000 * (60 * 60 * h + 60 * m + s) + ms


def move_forward(time):
    time_ms = parse_time_ms(time)
    time_ms = time_ms + time_to_move_forward_ms
    h = time_ms // (60 * 60 * 1000)
    minutes_ms = time_ms - h * (60 * 60 * 1000)
    m = minutes_ms // (60 * 1000)
    seconds_ms = minutes_ms - m * (60 * 1000)
    s = seconds_ms // 1000
    ms = seconds_ms - s * 1000
    ans = '{:02}:{:02}:{:02},{:03}'.format(h, m, s, ms)
    return ans


with open(filename, 'r', encoding="utf8") as inp:
    lines = inp.readlines()

new_lines = []
for line in lines:
    # print(line)
    times = [time for time in re.findall(r'(\d{2}:\d{2}:\d{2},\d+)', line)]
    assert len(times) == 0 or len(times) == 2
    moved_times = [move_forward(time) for time in times]
    for old, new in zip(times, moved_times):
        assert parse_time_ms(old) + time_to_move_forward_ms == parse_time_ms(new)
        line = line.replace(old, new)
    # print(line)
    new_lines.append(line)

new_filename = filename + '_moved.srt'
with open(new_filename, 'w', encoding='utf8') as otp:
    otp.writelines(new_lines)


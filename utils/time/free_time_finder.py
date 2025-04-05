DAYS = ['월', '화', '수', '목', '금']
HOURS = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]

def find_common_free_time(schedule_dicts):
    all_busy = {day: set() for day in DAYS}

    for schedule in schedule_dicts:
        for day in DAYS:
            all_busy[day].update(schedule.get(day, []))

    free_time = {day: [] for day in DAYS}
    for day in DAYS:
        for hour in HOURS:
            if hour not in all_busy[day]:
                free_time[day].append(hour)

    return free_time

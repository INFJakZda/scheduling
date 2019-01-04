import datetime
from math import floor

def prepareData(noFile, noInstance):
    with open("data/sch" + str(noFile) + ".txt", 'r') as fp:
        noInstances = int(fp.readline())
        arr = []
        for k in range(1, noInstances + 1):
            noLines = int(fp.readline())
            if (k == noInstance):
                for idx in range(noLines):
                    ele = list(map(int, fp.readline().split()))
                    ele.append(idx + 1)
                    arr.append(ele)
                break
            else:
                for _ in range(noLines):
                    fp.readline()
        return arr


def calculateSum(tasks, h):
    sum = 0
    for row in tasks:
        sum += row[0]
    return floor(sum * h)


def calculatePenalty(tasks, dueTime):
    penalty = 0
    time = 0
    for task in tasks:
        time += task[0]
        penaltyTime = dueTime - time
        if (penaltyTime < 0):
            penalty -= penaltyTime * task[2]
        elif (penaltyTime > 0):
            penalty += penaltyTime * task[1]
    return penalty

def schedule(tasks, dueTime):
    #TODO
    
    return 0, 0


if __name__ == '__main__':

    n = 100
    k = 8
    h = 0.2

    tasks = prepareData(n, k)
    dueTime = calculateSum(tasks, h)

    # Prepare data set
    tasks.sort(key=lambda task: task[1] - task[2])

    # Start timer
    startTime = datetime.datetime.now()

    # Schedule task with algorithm
    scheduledTasks, time = schedule(tasks, dueTime)

    # End timer
    endTime = datetime.datetime.now()
    diffTime = endTime - startTime

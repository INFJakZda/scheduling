import datetime
from math import floor

import numpy as np

def schedule(tasks, dueTime):
    #TODO
    return 0, 0

class GeneticAlgorithm:

    def __init__(self, population_size, num_generations, num_parents_mating, 
        offspring_size=-1, mutation_rate=0.25):
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        if offspring_size == -1:
            self.offspring_size = population_size - num_parents_mating
        else:
            self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate

    def loadInstance(self, n, k, h):

        def loadFromFile(noFile, noInstance):
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
                return np.asarray(arr, dtype=np.uint8)

        def calculateDueDate(tasks, h):
            return np.floor(np.sum(tasks[:, 0]) * h)

        self.tasks = loadFromFile(n, k)
        self.dueDate = calculateDueDate(self.tasks, h)

    def calculatePenalty(self, sequence):
        penalty = 0; time = 0
        for task_idx in sequence:
            task = self.tasks[task_idx]
            time += task[0]
            penaltyTime = self.dueDate - time
            if (penaltyTime < 0):
                penalty -= penaltyTime * task[2]
            elif (penaltyTime > 0):
                penalty += penaltyTime * task[1]
        return penalty

if __name__ == '__main__':

    n = 10
    k = 1
    h = 0.2

    # Start timer
    startTime = datetime.datetime.now()

    # Schedule task with algorithm
    # scheduledTasks, time = schedule(tasks, dueDate)

    GA = GeneticAlgorithm(population_size=10, num_generations=5, 
        num_parents_mating=5, offspring_size=5, mutation_rate=0.1)
    GA.loadInstance(n, k, h)


    # End timer
    endTime = datetime.datetime.now()
    diffTime = endTime - startTime

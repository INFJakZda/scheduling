import datetime
from math import floor
from datetime import timedelta
import sys

import numpy as np

def saveData(tasks, processTime, elapsedTime):
    times = [[] for _ in range(len(tasks))]
    time = 0
    for task in tasks:
        times[task[3] - 1] = time
        time += task[0]
    with open("result.txt", 'w') as fw:
        fw.write(str(processTime) + '\n')
        fw.write(str(elapsedTime) + '\n')
        fw.write(' '.join(map(str, times)))

class GeneticAlgorithm:

    def __init__(self, max_time, population_size, num_generations, 
        num_parents_mating, offspring_size=-1, mutation_rate=0.25):
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        if offspring_size == -1:
            self.offspring_size = population_size - num_parents_mating
        else:
            self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.max_time = max_time

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
                            ele.append(idx+1)
                            arr.append(ele)
                        break
                    else:
                        for _ in range(noLines):
                            fp.readline()
                return np.asarray(arr, dtype=np.uint32)

        def calculateDueDate(tasks, h):
            return np.floor(np.sum(tasks[:, 0]) * h)

        self.tasks = loadFromFile(n, k)
        self.dueDate = calculateDueDate(self.tasks, h)
        self.n = n #number of tasks

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

    def initializePopulation(self):
        population = np.zeros(shape=(self.population_size, self.n),
            dtype=np.uint32)
        for i in range(self.population_size):
            population[i, :] = np.random.permutation(self.n)
        return population

    def search(self):
        
        population = self.initializePopulation()

        # for i in range(self.num_generations):
        while(datetime.datetime.now() < self.max_time):
            
            # Measuring the fitness of each chromosome in the population.
            pop_scores = np.zeros(self.population_size, dtype=np.uint32)
            for p in range(self.population_size):
                pop_scores[p] = self.calculatePenalty(population[p, :])
            
            # Selecting the best parents in the population for mating.
            parents_to_mate = np.argsort(pop_scores)[:self.num_parents_mating]
            best_so_far = population[parents_to_mate[0], :]

            new_offspring = self.crossover(parents_to_mate, population)
            
            '''
            Here comes MUTATION over NEW_OFFSPRING...
            '''

            parents = population[parents_to_mate, :]
            population[:self.num_parents_mating, :] = parents
            population[self.num_parents_mating:, :] = new_offspring

        return best_so_far

    def crossover(self, parents_indices, population):
        offspring = np.zeros(shape=(self.offspring_size, self.n), dtype=np.uint32)
        num_parents = len(parents_indices)

        for k in range(self.offspring_size):
            # Indices of parents to mate.
            p1, p2 = parents_indices[k%num_parents], parents_indices[(k+1)%num_parents]
            
            parent1 = population[p1, :]
            parent2 = population[p2, :]

            child = self.mateParents(parent1, parent2)
            offspring[k, :] = child

        return offspring

    def mateParents(self, p1, p2):
        binary_string = np.random.randint(2, size=len(p1))
        while np.unique(binary_string).shape[0] == 1:
            binary_string = np.random.randint(2, size=len(p1))

        temp_p1 = p1 + 1
        temp_p2 = p2 + 1
        # p1 += 1; p2 += 1

        child = temp_p1 * binary_string
        p2_jobs_left = temp_p2[~np.in1d(temp_p2, child)]
        child[child == 0] = p2_jobs_left

        child -= 1
        # p1 -= 1; p2 -= 1

        return child

if __name__ == '__main__':

    n = 10
    k = 1
    h = 0.4
    c = 10
    reserveTime = 2 #miliseconds 10^-3

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        pass
    elif len(arguments) == 4:
        n = int(sys.argv[1])
        k = int(sys.argv[2])
        h = float(sys.argv[3])
        c = int(sys.argv[4])
    else:
        print("Usage: main.py n k h c")

    # Start timer
    startTime = datetime.datetime.now()
    maxTime = startTime + timedelta(milliseconds = c * n - reserveTime)

    population_size = 20
    num_generations = 10

    parents_mating_ratio = 0.5
    num_parents_mating = int(parents_mating_ratio * population_size)
    offspring_size = population_size - num_parents_mating

    mutation_rate = 0.1

    GA = GeneticAlgorithm(
        maxTime,
        population_size=population_size, 
        num_generations=num_generations, 
        num_parents_mating=num_parents_mating, 
        offspring_size=offspring_size, 
        mutation_rate=mutation_rate
    )
    GA.loadInstance(n, k, h)

    solution = GA.search()

    # End timer
    endTime = datetime.datetime.now()
    diffTime = endTime - startTime

    print("End of search!")
    print(diffTime)
    print("\nSolution:")
    print(solution)
    print(GA.calculatePenalty(solution))

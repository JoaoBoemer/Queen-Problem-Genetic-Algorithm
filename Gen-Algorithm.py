import random

def fitness_function(x):
    return x ** 2

def create_population(size):
    population = []
    for _ in range(size):
        individual = random.uniform(-100, 100)
        population.append(individual)
    return population

def select_parents(population, num_parents):
    parents = []
    for _ in range(num_parents):
        parent = random.choice(population)
        parents.append(parent)
    return parents

def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        child = (parent1 + parent2) / 2.0
        offspring.append(child)
    return offspring

def mutation(offspring, mutation_rate):
    mutated_offspring = []
    for child in offspring:
        if random.random() < mutation_rate:
            # Generate a random value in the range [-1, 1]
            mutation_value = random.uniform(-1, 1)
            child += mutation_value
        mutated_offspring.append(child)
    return mutated_offspring

def find_best_individual(population):
    best_fitness = float('-inf')
    best_individual = None
    for individual in population:
        fitness = fitness_function(individual)
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual
    return best_individual

# Genetic algorithm
def genetic_algorithm(population_size, num_generations, num_parents, offspring_size, mutation_rate):
    # Create initial population
    population = create_population(population_size)
    for generation in range(num_generations):
        # Select parents
        parents = select_parents(population, num_parents)

        # Create offspring through crossover
        offspring = crossover(parents, offspring_size)

        # Perform mutation on the offspring
        mutated_offspring = mutation(offspring, mutation_rate)

        # Replace the population with the new generation
        population = mutated_offspring

        # Find the best individual in the current population
        best_individual = find_best_individual(population)

        # Print the best individual in the current generation
        print(f"Generation {generation + 1}: Best Individual = {best_individual}")

    # Return the best individual after all generations
    return best_individual

# # Example usage
# population_size = 100
# num_generations = 50
# num_parents = 20
# offspring_size = 80
# mutation_rate = 0.1

# best_individual = genetic_algorithm(population_size, num_generations, num_parents, offspring_size, mutation_rate)
# print(f"\nBest Individual found: {best_individual}")
# print(f"Best Fitness: {fitness_function(best_individual)}")

import random

BOARD_SIZE = 9
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1

# Create an individual with a random chromosome
def create_individual():
    return random.sample(range(BOARD_SIZE), BOARD_SIZE)

# Create the initial population
def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

# Calculate the fitness of an individual
def calculate_fitness(individual):
    clashes = 0
    for i in range(BOARD_SIZE):
        for j in range(i + 1, BOARD_SIZE):
            if individual[i] == individual[j] or individual[i] - individual[j] == i - j or individual[i] - individual[j] == j - i:
                clashes += 1
    return clashes

# Select parents for reproduction using tournament selection
def select_parents(population):
    parents = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(population, 3)
        parent = min(tournament, key=calculate_fitness)
        parents.append(parent)
    return parents

# Perform crossover to create offspring
def crossover(parents):
    offspring = []
    for i in range(0, POPULATION_SIZE, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        crossover_point = random.randint(1, BOARD_SIZE - 2)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([child1, child2])
    return offspring

# Perform mutation on the offspring
def mutation(offspring):
    for i in range(len(offspring)):
        if random.random() < MUTATION_RATE:
            j = random.randint(0, BOARD_SIZE - 1)
            k = random.randint(0, BOARD_SIZE - 1)
            offspring[i][j], offspring[i][k] = offspring[i][k], offspring[i][j]
    return offspring

# Find the best individual in the population
def find_best_individual(population):
    return min(population, key=calculate_fitness)

# Genetic algorithm
def genetic_algorithm():
    population = create_population()

    for generation in range(NUM_GENERATIONS):
        parents = select_parents(population)
        offspring = crossover(parents)
        mutated_offspring = mutation(offspring)
        population = mutated_offspring

        best_individual = find_best_individual(population)
        print(f"Generation {generation + 1}: Best Individual = {best_individual}, Fitness = {calculate_fitness(best_individual)}")

    return best_individual

# Solve the Eight Queens Problem using the genetic algorithm
best_individual = genetic_algorithm()

print("\nSolution found:")
for i in range(BOARD_SIZE):
    row = ['Q' if j == best_individual[i] else '.' for j in range(BOARD_SIZE)]
    print(' '.join(row))
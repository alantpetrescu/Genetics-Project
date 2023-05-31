import copy
from matplotlib import pyplot as plt

from genetics_utils import *


# Functia pentru generarea candidatilor
def generate_candidates(dim: int, d: Tuple[float, float], precision: int, is_first_step: bool = True):
    list_candidates = np.random.uniform(d[0], d[1], dim)

    if is_first_step:
        file.write("Pasul 1: Selectam candidatii din populatie in mod aleatoriu\n")
        file.write("Populatia produsa: \n")
        for i in range(dim):
            list_candidates[i] = round(list_candidates[i], precision)
            file.write(f"{list_candidates[i]}\n")
        file.write("\n")

    return list_candidates


# Functia pentru codificarea candidatilor
def encode_candidates(list_candidates, d: Tuple[float, float], precision: int, is_first_step: bool = True) -> List[str]:
    start, end = d[0], d[1]
    nr_bits = math.ceil(math.log2((end - start) * (10 ** precision)))
    nr_intervals = 2 ** nr_bits
    delta = (end - start) / nr_intervals

    encoded_candidates = [encode_candidate(candidate, start, end, delta, nr_bits) for candidate in list_candidates]

    if is_first_step:
        file.write("Pasul 2: Codificam candidatii\n")
        file.write("Populatia codificata: \n\n")
        for i, encoded_candidate in enumerate(encoded_candidates):
            file.write(f"{list_candidates[i]} -> {encoded_candidate}\n")
        file.write("\n")

    return encoded_candidates


# functia pentru selectarea de tip turneu si elitista a candidatilor
def select_candidates(list_candidates: List[int], encoded_candidates: List[str], params: List[float],
                      is_first_step: bool = True) -> Tuple[List[int], List[str]]:
    # print(list_candidates)
    # print(encoded_candidates)
    list_fitness_candidates = get_fitness_list(list_candidates, params)
    #print("lista de fitness: " + str(list_fitness_candidates))
    best_candidate_index = give_best_candidate_index(list_fitness_candidates)
    #print("id-ul celui mai bunt candidat in functie de fitness: " + str(best_candidate_index))

    n = len(list_candidates)
    for i in range(1, n):
        list_fitness_candidates[i] += list_fitness_candidates[i - 1]
    # print(list_fitness_candidates)
    fitness_sum = list_fitness_candidates[n - 1]
    probability_distribution = [fitness_candidate / fitness_sum for fitness_candidate in list_fitness_candidates]
    # print(probability_distribution)

    if is_first_step:
        file.write("Pasul 3: Selectia candidatilor\n")
        file.write("Populatia inainte de selectie (valoarea cromozomului -> cromozomul codificat -> valoarea functiei de fitness -> probabilitatea de distributie -> probabilitatea acumulata): \n\n")
        for i, encoded_candidate in enumerate(encoded_candidates):
            if i == 0:
                file.write(f"{list_candidates[i]} -> {encoded_candidate} -> {calculate_function(list_candidates[i], params)} -> {probability_distribution[i]} -> {probability_distribution[i]}\n")
            else:
                file.write(f"{list_candidates[i]} -> {encoded_candidate} -> {calculate_function(list_candidates[i], params)} -> {probability_distribution[i] - probability_distribution[i - 1]} -> {probability_distribution[i]}\n")
        file.write("\n")

    new_candidates = [0] * n
    new_encoded_candidates = [''] * n
    for i in range(n - 1):
        u = np.random.uniform()
        # print(u)
        # print(binary_search(probability_distribution, u))
        searched_i = binary_search(probability_distribution, u)
        new_candidates[i] = list_candidates[searched_i]
        new_encoded_candidates[i] = encoded_candidates[searched_i]

    new_candidates[n - 1] = list_candidates[best_candidate_index]
    new_encoded_candidates[n - 1] = encoded_candidates[best_candidate_index]

    if is_first_step:
        file.write("Populatia dupa selectie (valoarea cromozomului -> cromozomul codificat -> valoarea functiei de fitness -> probabilitatea de distributie -> probabilitatea acumulata): \n\n")
        for i in range(n):
            if i == 0:
                file.write(f"{new_candidates[i]} -> {new_encoded_candidates[i]} -> {calculate_function(new_candidates[i], params)} -> {probability_distribution[i]} -> {probability_distribution[i]}\n")
            else:
                file.write(f"{new_candidates[i]} -> {new_encoded_candidates[i]} -> {calculate_function(new_candidates[i], params)} -> {probability_distribution[i] - probability_distribution[i - 1]} -> {probability_distribution[i]}\n")
        file.write("\n")

    return new_candidates, new_encoded_candidates


# functia pentru incrucisarea candidatilor
def crossover_candidates(list_candidates: List[float], encoded_candidates: List[str], crossover_probability: float, start: float, delta: float, precision: int, is_first_step: bool = True) -> Tuple[List[float], List[str]]:
    #print("list of encoded candidates: " + str(encoded_candidates))
    is_marked_candidates = give_marked_candidates(len(encoded_candidates), crossover_probability)
    #print("list of the is_marked_candidates list: " + str(is_marked_candidates))
    n = len(is_marked_candidates)

    marked_indexes = [-1] * len(list(filter(lambda x: x is True, is_marked_candidates)))
    m, i, j = len(marked_indexes), 0, 0

    while i < n:
        if is_marked_candidates[i]:
            marked_indexes[j] = i
            j += 1
        i += 1

    if is_first_step:
        file.write("Pasul 4: incrucisam candidatii din populatie\n")
        file.write("Perechile incrucisate:\n\n")
    #print("list of marked candidates' indexes: " + str(marked_indexes))
    np.random.shuffle(marked_indexes)
    #print("shuffled list of marked candidates' indexes" + str(marked_indexes))

    m -= m % 2
    for i in range(0, m, 2):
        p, q = marked_indexes[i], marked_indexes[i + 1]
        list_candidates[p], list_candidates[q], encoded_candidates[p], encoded_candidates[q] = crossover_pair(file, encoded_candidates[p], encoded_candidates[q], list_candidates[p], list_candidates[q], start, delta, precision, is_first_step)

    if is_first_step:
        file.write("Populatia dupa incrucisare:\n")
        for i in range(n):
            file.write(f"{encoded_candidates[i]}   ->   {list_candidates[i]}\n")

    return list_candidates, encoded_candidates


# functia pentru mutatia candidatilor
def mutate_candidates(candidates: List[float], encoded_candidates: List[str], mutation_probability: float, start: float, delta: float, precision: int, is_first_step: bool = True) -> Tuple[List[float], List[str]]:
    if is_first_step:
        file.write("Pasul 5: Mutam candidatii\n\n")
    mutated_candidates, mutated_encoded_candidates = unzip([mutate_candidate(file, candidate, encoded_candidate, mutation_probability, start, delta, precision, is_first_step) for candidate, encoded_candidate in zip(candidates, encoded_candidates)])

    if is_first_step:
        file.write("\nPopulatia mutata:\n")
        n = len(candidates)
        for i in range(n):
            file.write(f"{mutated_encoded_candidates[i]}   ->   {mutated_candidates[i]}\n")

    return mutated_candidates, mutated_encoded_candidates


def find_best_candidate(dim: int, d: Tuple[float, float], params: List[float], precision: int,
                        crossover_probability: float, mutation_probability: float, nr_steps: int):
    delta = give_delta(d, precision)

    # Datele despre algoritmul de genetica
    file.write("Avem urmatoarele date: \n")
    file.write("---------------------------------------------------------------------------\n")
    file.write(f"Dimensiunea populatiei: {dim}\n")
    file.write(f"Domeniul de definitie al functiei: [{d[0]}, {d[1]}]\n")
    file.write(f"Forma functiei de gradul 2: {params[0]}*a^2 + {params[1]}*a + {params[2]}\n")
    file.write(f"Precizia cu care se lucreaza: {precision}\n")
    file.write(f"Probabilitatea de incrucisare: {crossover_probability}\n")
    file.write(f"Probabilitatea de mutatie: {mutation_probability}\n")
    file.write(f"Numarul de etape al algoritmului: {nr_steps}\n")
    file.write("---------------------------------------------------------------------------\n\n\n")

    list_candidates, encoded_candidates = [], []
    list_max_fitness, list_mean_fitness = [], []

    for i in range(nr_steps):
        # etapa i
        file.write(f"etapa {i + 1}:\n")
        file.write("---------------------------------------------------------------------------\n")

        # pasul 1: selectarea candidatilor/cromozomilor
        if i == 0:
            is_first_step = True
            list_candidates = list(generate_candidates(dim, d, precision))
            encoded_candidates = encode_candidates(list_candidates, d, precision, is_first_step)
        else:
            is_first_step = False
            list_fitness = get_fitness_list(list_candidates, params)
            file.write(f"Max fitness: {max(list_fitness)}\n")
            file.write(f"Mean fitness: {get_mean(list_fitness)}\n")

        list_fitness = get_fitness_list(list_candidates, params)
        list_max_fitness.append(max(list_fitness))
        list_mean_fitness.append(get_mean(list_fitness))
        # pasul 3: selectarea de tip ruleta si elitista
        new_list_candidates, new_encoded_candidates = select_candidates(list_candidates, encoded_candidates, params, is_first_step)

        # pasul 4: incrucisarea
        crossovered_candidates, crossovered_encoded_candidates = crossover_candidates(new_list_candidates[:dim - 1], new_encoded_candidates[:dim - 1], crossover_probability, d[0], delta, precision, is_first_step)
        crossovered_candidates.append(new_list_candidates[dim - 1])
        crossovered_encoded_candidates.append(new_encoded_candidates[dim - 1])
        if is_first_step:
            file.write(f"{crossovered_encoded_candidates[dim - 1]}   ->   {crossovered_candidates[dim - 1]}\n\n")

        # pasul 5: mutatia
        mutated_candidates, mutated_crossovered_candidates = mutate_candidates(crossovered_candidates[:dim - 1], crossovered_encoded_candidates[:dim - 1], mutation_probability, d[0], delta, precision, is_first_step)
        mutated_candidates.append(crossovered_candidates[dim - 1])
        mutated_crossovered_candidates.append(crossovered_encoded_candidates[dim - 1])
        if is_first_step:
            file.write(f"{mutated_crossovered_candidates[dim - 1]}   ->   {mutated_candidates[dim - 1]}\n")

        list_candidates = copy.deepcopy(mutated_candidates)
        file.write("---------------------------------------------------------------------------\n\n")

    file.write("Populatia finala:\n")

    list_fitness = get_fitness_list(list_candidates, params)
    list_max_fitness.append(max(list_fitness))
    list_mean_fitness.append(get_mean(list_fitness))

    file.write(f"Max fitness: {max(list_fitness)}\n")
    file.write(f"Mean fitness: {get_mean(list_fitness)}\n")

    plt.plot(list(range(nr_steps + 1)), list_max_fitness, label = "Max Fitness")
    plt.plot(list(range(nr_steps + 1)), list_mean_fitness, label = "Mean Fitness")
    plt.xlabel("Generation")
    plt.legend()
    plt.show()

    return list_candidates


if __name__ == '__main__':
    file = open("Evolutie.txt", "w")
    find_best_candidate(20, (-1, 2), [16, 4, 20], 6, 0.25, 0.1, 50)

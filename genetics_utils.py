from typing import Tuple, List, Any
import numpy as np
import math


def encode_candidate(candidate: float, start: float, end: float, delta: float, nr_bits: int) -> str:
    index_of_interval = get_index_of_interval(candidate, start, end, delta)
    binary_list = []

    while index_of_interval:
        binary_list.append(index_of_interval & 1)
        index_of_interval >>= 1

    n = len(binary_list)

    for _ in range(nr_bits - n):
        binary_list.append(1)

    binary_list.reverse()
    return ''.join(map(str, binary_list))


def give_delta(d: Tuple[float, float], precision: int):
    start, end = d[0], d[1]
    nr_bits = math.ceil(math.log2((end - start) * (10 ** precision)))
    nr_intervals = 2 ** nr_bits
    delta = (end - start) / nr_intervals

    return delta


def get_index_of_interval(candidate: float, start: float, end: float, delta: float) -> int:
    if candidate == end:
        return math.floor((end - start) // delta) - 1

    return math.floor((candidate - start) // delta)


def calculate_function(x: float, params: List[float]) -> float:
    a, b, c, d = params

    return a * x ** 3 + b * x ** 2 + c * x + d


def binary_search(l: List[Any], elem: Any):
    st, dr = 0, len(l) - 1

    while st <= dr:
        mij = (st + dr) // 2

        if l[mij] <= elem:
            st = mij + 1
        else:
            dr = mij - 1

    return min(st, len(l) - 1)


def give_best_candidate_index(list_candidates: List[float]) -> int:
    best_candidate, n = 0, len(list_candidates)
    best_candidate_index = -1

    for i in range(0, n):
        if best_candidate < list_candidates[i]:
            best_candidate = list_candidates[i]
            best_candidate_index = i

    return best_candidate_index


def select_or_not_element(probability: float) -> bool:
    uniform_probability = np.random.uniform()

    return uniform_probability < probability


def give_marked_candidates(n: int, probability: float) -> List[bool]:
    return [select_or_not_element(probability) for _ in range(n)]


def binary_to_decimal(binary_list: List[int]) -> int:
    n, x, p2 = len(binary_list), 0, 1

    for i in range(n - 1, -1, -1):
        x += binary_list[i] * p2
        p2 *= 2

    return x


def give_number_from_interval(nr_interval: int, start: float, delta: float, precision: int) -> float:
    return round(start + nr_interval * delta, precision)


def unzip(list_total: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
    list1, list2 = [], []
    n = len(list_total)

    for i in range(n):
        list1.append(list_total[i][0])
        list2.append(list_total[i][1])

    return list1, list2


def cross_pair(file, candidate1: str, candidate2: str, x1: float, x2: float, cut_point: int, start: float, delta: float, precision: int, is_first_step: bool = True) -> Tuple[float, float, str, str]:
    if is_first_step:
        file.write(f"Inainte de incrucisare:   ")
        file.write(f"{candidate1} ({x1})     {candidate2} ({x2}) in punctul de taiere {cut_point}\n")

    binary_list1, binary_list2 = list(map(int, candidate1)), list(map(int, candidate2))

    for i in range(cut_point):
        binary_list1[i], binary_list2[i] = binary_list2[i], binary_list1[i]

    new_x1 = give_number_from_interval(binary_to_decimal(binary_list1), start, delta, precision)
    new_x2 = give_number_from_interval(binary_to_decimal(binary_list2), start, delta, precision)
    #print(binary_list1, binary_to_decimal(binary_list1), new_x1, start, delta)
    crossed_candidate1, crossed_candidate2 = ''.join(map(str, binary_list1)), ''.join(map(str, binary_list2))
    if is_first_step:
        file.write("Dupa incrucisare:   ")
        file.write(f"{candidate1} ({new_x1})     {candidate2} ({new_x2})\n\n")

    return new_x1, new_x2, crossed_candidate1, crossed_candidate2


def crossover_pair(file, encoded_candidate1: str, encoded_candidate2: str, candidate1: float, candidate2: float, start: float, delta: float, precision: int, is_first_step: bool = True) -> Tuple[float, float, str, str]:
    n, m = len(encoded_candidate1), len(encoded_candidate2)

    if n is not m:
        raise Exception("The pair is not formed of candidates of equal lengths")

    cut_point = np.random.randint(0, n)

    return cross_pair(file, encoded_candidate1, encoded_candidate2, candidate1, candidate2, cut_point, start, delta, precision, is_first_step)


def mutate_gene(file, pos: int, gene: int, mutation_probability: float, is_first_step: bool = True) -> int:
    u = np.random.uniform()

    if u < mutation_probability:
        if is_first_step:
            file.write(f"{pos} ")
        #print("has mutated")
        gene ^= 1

    return gene


def mutate_candidate(file, candidate: float, encoded_candidate: str, mutation_probability: float, start: float, delta: float, precision: int, is_first_step: bool = True) -> Tuple[float, str]:
    if is_first_step:
        file.write(f"Mutam candidatul {encoded_candidate} ({candidate})\n")
        file.write("Lista de gene mutate dupa pozitie: ")

    mutated_encoded_candidate = [mutate_gene(file, pos, int(gene), mutation_probability, is_first_step) for pos, gene in enumerate(encoded_candidate)]

    if is_first_step:
        if ''.join(map(str, mutated_encoded_candidate)) == encoded_candidate:
            file.write("nimic\n")
        else:
            file.write("\n")
    mutated_candidate = give_number_from_interval(binary_to_decimal(mutated_encoded_candidate), start, delta, precision)
    #print(mutated_encoded_candidate, binary_to_decimal(mutated_encoded_candidate), mutated_candidate, start, delta)
    mutated_encoded_candidate = ''.join(map(str, mutated_encoded_candidate))
    return mutated_candidate, mutated_encoded_candidate


def get_fitness_list(list_candidates: List[float], params: List[float]) -> List[float]:
    return [calculate_function(list_candidate, params) for list_candidate in list_candidates]


def get_mean(list_fitness: List[float]) -> float:
    return sum(list_fitness) / len(list_fitness)

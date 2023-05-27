"""This module contains functions for sorting loops"""
from collections import deque
from numpy import (
    ndarray,
    array,
    zeros,
    argmin,
)


def get_n_loop(loop: list) -> list[tuple[int, str, str]]:
    """Converts a loop to a list of tuples with the index and the color"""
    return [(i,) + tpl[1:] for i, tpl in enumerate(loop)]


def generate_distance_matrix(n_loop: list, func: callable, loop_length: int) -> ndarray:
    """Generates a distance matrix for a loop"""
    distance_matrix: ndarray = zeros((loop_length, loop_length))
    for i in range(loop_length):
        for j in range(i):
            distance_matrix[i][j] = distance_matrix[j][i] = func(n_loop[i][1], n_loop[j][1])
    return distance_matrix


def resort_loop(loop: list[tuple[int, int, int]], func: callable,
                loop_length: int) -> list[tuple[int, int, int]]:
    """Reorders a loop to minimize the distance between the colors"""
    n_loop = deque(get_n_loop(loop))
    distance_matrix = generate_distance_matrix(n_loop, func, loop_length)
    counter = 0
    while counter < 100000:
        counter += 1
        moving_loop_entry = n_loop.pop()
        n_loop = move_entry(n_loop, moving_loop_entry, distance_matrix)
        if n_loop[0][0] == 0:
            break
    return [(loop[tpl[0]][0],) + tpl[1:] for tpl in n_loop]


def move_entry(loop: deque[tuple[int, int, int]],
               moving_loop_entry: tuple[int, int, int],
               distance_matrix: ndarray) -> deque[tuple[int, int, int]]:
    """Moves the entry with the least average distance to the front of the loop"""
    behind_indices = array([loop[i - 1][0] for i in range(1, len(loop) - 1)])
    ahead_indices = array([loop[i + 1][0] for i in range(len(loop) - 2)])
    behind_distances = distance_matrix[behind_indices, moving_loop_entry[0]]
    ahead_distances = distance_matrix[ahead_indices, moving_loop_entry[0]]
    avg_of_distances = (behind_distances + ahead_distances) / 2
    min_index = argmin(avg_of_distances)
    if min_index == len(loop) - 3:
        loop.appendleft(moving_loop_entry)
    else:
        loop.rotate(-(min_index + 1))
        loop.appendleft(moving_loop_entry)
        loop.rotate(min_index + 1)
    return loop


def loop_sort(entries: list[tuple], func: callable) -> list[tuple]:
    """Sorts a list of entries by the function func"""
    loop: deque = deque([entries[0]])
    entries: deque = deque(entries[1:])
    length: int = len(entries)
    for _ in range(1, length + 1):
        item_one: tuple = loop[-1]
        item_two: deque = entries
        j: int = find_minimum(item_one, func, item_two)
        loop.append(item_two[j])
        item_two.rotate(-j)
        item_two.popleft()
    return list(loop)


def find_minimum(p_entry: tuple, func: callable, q_entries: tuple) -> int:
    """Finds the value of q_entries that minimizes the function func(p_entry, q_entry)"""
    return (min(enumerate(q_entries), key=lambda x: func(p_entry[1], x[1][1])))[0]

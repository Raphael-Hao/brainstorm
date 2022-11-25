# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import itertools

def sorted_k_partitions(seq, k, length):
    """Returns a list of all unique k-partitions of `seq`.

    Each partition is a list of parts, and each part is a tuple.

    The parts in each individual partition will be sorted in shortlex
    order (i.e., by length first, then lexicographically).

    The overall list of partitions will then be sorted by the length
    of their first part, the length of their second part, ...,
    the length of their last part, and then lexicographically.
    """
    n = len(seq)
    groups = []  # a list of lists, currently empty

    def generate_partitions(i):
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > (k - len(groups)) * length:
                for group in groups:
                    if len(group) < length:
                        group.append(seq[i])
                        yield from generate_partitions(i + 1)
                        group.pop()
                    # else:
                    #     group.append(seq[i])
                    #     yield from generate_partitions(i + 1)
                    #     group.pop()

            if len(groups) < k:
                groups.append([seq[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)

    # Sort the parts in each partition in shortlex order
    result = [sorted(ps, key=lambda p: (len(p), p)) for ps in result]
    # Sort partitions by the length of each part, then lexicographically.
    result = sorted(result, key=lambda ps: (*map(len, ps), ps))

    all_ordered_partitions = []

    for partition in result:
        all_ordered_partitions.extend(list(itertools.permutations(partition)))

    return all_ordered_partitions


#%%
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(sorted_k_partitions(a, 2, 5))
# %%

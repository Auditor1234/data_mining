#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to itemsets.
"""

import itertools
import numbers
import typing
import collections
from dataclasses import field, dataclass
import collections.abc


@dataclass
class ItemsetCount:
    itemset_count: int = 0
    members: set = field(default_factory=set)


class TransactionManager:
    def __init__(self, transactions: typing.Iterable[typing.Iterable[typing.Hashable]]):
        self.indices_by_item = collections.defaultdict(set)
        i = -1
        for i, transaction in enumerate(transactions):
            for item in transaction:
                self.indices_by_item[item].add(i)

        self._transactions = i + 1

    @property
    def items(self):
        return set(self.indices_by_item.keys())

    def __len__(self):
        return self._transactions

    def transaction_indices(self, transaction: typing.Iterable[typing.Hashable]):
        transaction = set(transaction) 
        item = transaction.pop()
        indices = self.indices_by_item[item]
        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self.indices_by_item[item])
        return indices

    def transaction_indices_sc(self, transaction: typing.Iterable[typing.Hashable], min_support: float = 0):
        transaction = sorted(transaction, key=lambda item: len(self.indices_by_item[item]), reverse=True)

        item = transaction.pop()
        indices = self.indices_by_item[item]
        support = len(indices) / len(self)
        if support < min_support:
            return False, None

        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self.indices_by_item[item])
            support = len(indices) / len(self)
            if support < min_support:
                return False, None

        return True, indices


def join_step(itemsets: typing.List[tuple]):

    i = 0
    while i < len(itemsets):
        skip = 1

        *itemset_first, itemset_last = itemsets[i]

        tail_items = [itemset_last]
        tail_items_append = tail_items.append

        for j in range(i + 1, len(itemsets)):
            *itemset_n_first, itemset_n_last = itemsets[j]

            if itemset_first == itemset_n_first:
                tail_items_append(itemset_n_last)
                skip += 1

            else:
                break
        itemset_first_tuple = tuple(itemset_first)
        for a, b in itertools.combinations(tail_items, 2):
            yield itemset_first_tuple + (a,) + (b,)

        i += skip


def prune_step(itemsets: typing.Iterable[tuple], possible_itemsets: typing.List[tuple]):
    itemsets = set(itemsets)

    for possible_itemset in possible_itemsets:
        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1 :]

            if removed not in itemsets:
                break

        else:
            yield possible_itemset


def apriori_gen(itemsets: typing.List[tuple]):
    possible_extensions = join_step(itemsets)
    yield from prune_step(itemsets, possible_extensions)


def itemsets_from_transactions(
    transactions: typing.Iterable[typing.Union[set, tuple, list]],
    min_support: float,
    max_length: int = 8,
):

    manager = TransactionManager(transactions)

    transaction_count = len(manager)
    if transaction_count == 0:
        return dict(), 0


    candidates: typing.Dict[tuple, int] = {(item,): len(indices) for item, indices in manager.indices_by_item.items()}
    large_itemsets: typing.Dict[int, typing.Dict[tuple, int]] = {
        1: {item: count for (item, count) in candidates.items() if (count / len(manager)) >= min_support}
    }

    if not large_itemsets.get(1, dict()):
        return dict(), 0

    k = 2
    while large_itemsets[k - 1] and (max_length != 1):
        itemsets_list = sorted(item for item in large_itemsets[k - 1].keys())
        C_k: typing.List[tuple] = list(apriori_gen(itemsets_list))

        if not C_k:
            break

        found_itemsets: typing.Dict[tuple, int] = dict()
        for candidate in C_k:
            over_min_support, indices = manager.transaction_indices_sc(candidate, min_support=min_support)
            if over_min_support:
                found_itemsets[candidate] = len(indices)

        if not found_itemsets:
            break

        large_itemsets[k] = {i: counts for (i, counts) in found_itemsets.items()}
        k += 1
        if k > max_length:
            break

    itemsets_out = {
        length: {
            item: ItemsetCount(itemset_count=count, members=manager.transaction_indices(set(item)))
            for (item, count) in itemsets.items()
        }
        for (length, itemsets) in large_itemsets.items()
    }
    return itemsets_out, len(manager)
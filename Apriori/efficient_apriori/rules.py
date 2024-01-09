#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to association rules.
"""

import typing
import numbers
import itertools
from efficient_apriori.itemsets import apriori_gen


class Rule:
    def __init__(
        self,
        lhs: tuple,
        rhs: tuple,
        count_full: int = 0,
        count_lhs: int = 0,
        count_rhs: int = 0,
        num_transactions: int = 0,
    ):

        self.lhs = lhs 
        self.rhs = rhs
        self.count_full = count_full
        self.count_lhs = count_lhs
        self.count_rhs = count_rhs
        self.num_transactions = num_transactions

    @property
    def confidence(self):
        return self.count_full / self.count_lhs

    @property
    def support(self):
        return self.count_full / self.num_transactions

    @staticmethod
    def _pf(s):
        return "{" + ", ".join(str(k) for k in s) + "}"

    def __repr__(self):
        return "{} -> {}".format(self._pf(self.lhs), self._pf(self.rhs))

    def __str__(self):
        conf = "conf: {0:.3f}".format(self.confidence)
        supp = "supp: {0:.3f}".format(self.support)

        return "{} -> {} ({}, {})".format(self._pf(self.lhs), self._pf(self.rhs), conf, supp)

def generate_rules_simple(
    itemsets: typing.Dict[int, typing.Dict],
    min_confidence: float,
    num_transactions: int,
):
    """
    DO NOT USE. This is a simple top-down algorithm for generating association
    rules. It is included here for testing purposes, and because it is
    mentioned in the 1994 paper by Agrawal et al. It is slow because it does
    not enumerate the search space efficiently: it produces duplicates, and it
    does not prune the search space efficiently.

    Simple algorithm for generating association rules from itemsets.
    """

    # Iterate over every size
    for size in itemsets.keys():
        # Do not consider itemsets of size 1
        if size < 2:
            continue

        # This algorithm returns duplicates, so we keep track of items yielded
        # in a set to avoid yielding duplicates
        yielded: set = set()
        yielded_add = yielded.add

        # Iterate over every itemset of the prescribed size
        for itemset in itemsets[size].keys():
            # Generate rules
            for result in _genrules(itemset, itemset, itemsets, min_confidence, num_transactions):
                # If the rule has been yieded, keep going, else add and yield
                if result in yielded:
                    continue

                yielded_add(result)
                yield result


def _genrules(l_k, a_m, itemsets, min_conf, num_transactions):
    """
    DO NOT USE. This is the gen-rules algorithm from the 1994 paper by Agrawal
    et al. It's a subroutine called by `generate_rules_simple`. However, the
    algorithm `generate_rules_simple` should not be used.
    The naive algorithm from the original paper.

    Parameters
    ----------
    l_k : tuple
        The itemset containing all elements to be considered for a rule.
    a_m : tuple
        The itemset to take m-length combinations of, an move to the left of
        l_k. The itemset a_m is a subset of l_k.
    """

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    # Iterate over every k - 1 combination of a_m to produce
    # rules of the form a -> (l - a)
    for a_m in itertools.combinations(a_m, len(a_m) - 1):
        # Compute the count of this rule, which is a_m -> (l_k - a_m)
        confidence = count(l_k) / count(a_m)

        # Keep going if the confidence level is too low
        if confidence < min_conf:
            continue

        # Create the right hand set: rhs = (l_k - a_m) , and keep it sorted
        rhs = set(l_k).difference(set(a_m))
        rhs = tuple(sorted(rhs))

        # Create new rule object and yield it
        yield Rule(a_m, rhs, count(l_k), count(a_m), count(rhs), num_transactions)

        # If the left hand side has one item only, do not recurse the function
        if len(a_m) <= 1:
            continue
        yield from _genrules(l_k, a_m, itemsets, min_conf, num_transactions)


def generate_rules_apriori(
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_confidence: float,
    num_transactions: int,
):
    def count(itemset):
        return itemsets[len(itemset)][itemset]

    for size in itemsets.keys():
        if size < 2:
            continue

        for itemset in itemsets[size].keys():
            H_1 = []
            for removed in itertools.combinations(itemset, 1):
                remaining = set(itemset).difference(set(removed))
                lhs = tuple(sorted(remaining))

                conf = count(itemset) / count(lhs)
                if conf >= min_confidence:
                    yield Rule(
                        lhs,
                        removed,
                        count(itemset),
                        count(lhs),
                        count(removed),
                        num_transactions,
                    )

                    H_1.append(removed)

            if len(H_1) == 0:
                continue

            yield from _ap_genrules(itemset, H_1, itemsets, min_confidence, num_transactions)


def _ap_genrules(
    itemset: tuple,
    H_m: typing.List[tuple],
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_conf: float,
    num_transactions: int,
):
    def count(itemset):
        return itemsets[len(itemset)][itemset]

    if len(itemset) <= (len(H_m[0]) + 1):
        return

    H_m = list(apriori_gen(H_m))
    H_m_copy = H_m.copy()

    for h_m in H_m:
        lhs = tuple(sorted(set(itemset).difference(set(h_m))))

        if (count(itemset) / count(lhs)) >= min_conf:
            yield Rule(
                lhs,
                h_m,
                count(itemset),
                count(lhs),
                count(h_m),
                num_transactions,
            )
        else:
            H_m_copy.remove(h_m)

    if H_m_copy:
        yield from _ap_genrules(itemset, H_m_copy, itemsets, min_conf, num_transactions)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "-v"])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level implementations of the apriori algorithm.
"""

import typing
from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import generate_rules_apriori


def apriori(
    transactions: typing.Iterable[typing.Union[set, tuple, list]],
    min_support: float = 0.5,
    min_confidence: float = 0.5,
    max_length: int = 8,
):
    itemsets, num_trans = itemsets_from_transactions(
        transactions,
        min_support,
        max_length,
    )

    itemsets_raw = {
        length: {item: counter.itemset_count for (item, counter) in itemsets.items()}
        for (length, itemsets) in itemsets.items()
    }
    rules = generate_rules_apriori(itemsets_raw, min_confidence, num_trans)

    return itemsets, list(rules)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "-v"])

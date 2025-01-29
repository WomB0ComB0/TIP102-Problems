# pylint: disable=W0611
from typing import List, Dict, Tuple, Any, Set, Union, Optional, TypeVar, Generic
import random
import string
from collections import Counter, defaultdict, deque
from itertools import product, permutations, combinations
from functools import lru_cache, cache
import re
import math

from utils.data_generators import DataGenerators
from dsa.nodes import SinglyLinkedListNode

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Utils(DataGenerators):
    @staticmethod
    def flatten_list(nested_list: List[List[int]]) -> List[int]:
        """Flattens a nested list of integers."""
        return [item for sublist in nested_list for item in sublist]

    @staticmethod
    def invert_dict(d: Dict[int, int]) -> Dict[int, int]:
        """Inverts a dictionary (keys become values and values become keys)."""
        return {v: k for k, v in d.items()}

    @staticmethod
    def random_string(length: int) -> str:
        """Generates a random string of a given length."""
        return "".join(random.choices(string.ascii_lowercase, k=length))

    @staticmethod
    def prime_numbers(n: int) -> List[int]:
        """Generates a list of the first n prime numbers."""
        primes: List[int] = []
        candidate: int = 2
        while len(primes) < n:
            for prime in primes:
                if candidate % prime == 0:
                    break
            else:
                primes.append(candidate)
            candidate += 1
        return primes

    @staticmethod
    def fibonacci(n: int) -> List[int]:
        """Generates a list of the first n Fibonacci numbers."""
        fibs: List[int] = [0, 1]
        for _ in range(2, n):
            fibs.append(fibs[-1] + fibs[-2])
        return fibs[:n]

    @staticmethod
    def is_palindrome(s: str) -> bool:
        """Checks if a string is a palindrome."""
        return s == s[::-1]

    @staticmethod
    def is_anagram(s: str, t: str) -> bool:
        """Checks if two strings are anagrams of each other."""
        return Counter(s) == Counter(t)

    @staticmethod
    def is_prime(n: int) -> bool:
        """Checks if a number is prime."""
        if n == 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def is_power_of_two(n: int) -> bool:
        """Checks if a number is a power of two."""
        return n > 0 and n & (n - 1) == 0

    @staticmethod
    def is_power_of_three(n: int) -> bool:
        """Checks if a number is a power of three."""
        return n > 0 and 3**19 % n == 0

    @staticmethod
    def subsets_brute(nums: List[Any]) -> List[List[Any]]:
        result: List[List[Any]] = []
        for i in range(1 << len(nums)):
            subset: List[Any] = []
            for j in range(len(nums)):
                if i & (1 << j):
                    subset.append(nums[j])
            result.append(subset)
        return result

    @staticmethod
    def subsets_optimal(list1: List[int]) -> List[List[int]]:
        result: List[List[int]] = [[]]
        for num in list1:
            result += [i + [num] for i in result]
        return result

    @staticmethod
    def find_middle_node(head: SinglyLinkedListNode) -> int:
        """
        Write a function find_middle_node() that takes in the head of a
        linked list and returns the "middle" node.

        - If the linked list has an even length and there are two "middle" nodes, return the first middle node.
        - (E.g., "1 -> 2 -> 3 -> 4" would return 2, not 3.)
        """
        if not head:
            return head
        slow: SinglyLinkedListNode = head
        fast: SinglyLinkedListNode = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow.value

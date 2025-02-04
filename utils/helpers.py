import collections
from typing import List, Any, Dict, Set, Optional


class Height:  # Creating a height class
    def __init__(self) -> None:  # Initializing the height
        self.height: int = 0


def bfs(graph: Dict[Any, List[Any]], root: Any) -> None:  # BFS algorithm
    visited, queue = set(), collections.deque([root])
    visited.add(root)
    while queue:
        # Dequeue a vertex from queue
        vertex: Any = queue.popleft()
        print(str(vertex) + " ", end="")
        # If not visited, mark it as visited, and enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


def dfs(graph: Dict[Any, List[Any]], start: Any) -> Set[Any]:  # DFS algorithm
    visited, stack = set(), [start]
    while stack:
        vertex: Any = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited


def partition(
    array: List[Any], low: int, high: int
) -> int:  # function to find the partition position
    # choose the rightmost element as pivot
    pivot: Any = array[high]
    # pointer for greater element
    i: int = low - 1
    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:
            # if element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1
            # swapping element at i with element at j
            (array[i], array[j]) = (array[j], array[i])
    # swap the pivot element with the greater element specified by i
    (array[i + 1], array[high]) = (array[high], array[i + 1])
    # return the position from where partition is done
    return i + 1


class Stack:
    def __init__(self):  # Initializing the stack
        self.stack: List[Any] = []

    def create_stack(self) -> List[Any]:  # Creating a stack
        stack: List[Any] = []
        return stack

    def check_empty(self) -> bool:  # Creating an empty stack
        return len(self.stack) == 0

    def push(self, item: Any) -> None:  # Adding items into the stack
        self.stack.append(item)
        print("pushed item: " + item)

    def pop(self) -> Any:  # Removing an element from the stack
        if self.check_empty():
            return "stack is empty"
        return self.stack.pop()


class HashTable:
    def __init__(self, MAX: int = 10, arr: List[Any] = None):
        self.MAX = MAX
        if arr is not None:
            self.arr: List[Any] = arr
        else:
            self.arr: List[Any] = [None for i in range(self.MAX)]

    def checkPrime(self, n: int) -> int:  # Check if a number is prime
        if n == 1 or n == 0:
            return 0
        for i in range(2, n // 2):
            if n % i == 0:
                return 0
        return 1

    def getPrime(
        self, n: int
    ) -> int:  # Get the prime number just greater than the given number
        if n % 2 == 0:
            n = n + 1
        while not self.checkPrime(n):
            n += 2
        return n

    def hashFunction(self, key: Any) -> int:  # Remainder Method
        capacity: int = self.getPrime(self.MAX)
        return key % capacity

    def insertData(
        self, key: Any, data: Any
    ) -> None:  # Insert "data" into hash table at "key" index
        index: int = self.hashFunction(key)
        self.arr[index] = [key, data]

    def removeData(self, key: Any) -> None:  # Remove a key from the hash table
        index: int = self.hashFunction(key)
        self.arr[index] = 0


class Node:  # Creating a node
    def __init__(self, item: Any) -> None:
        self.item: Any = item
        self.next: Optional[Node] = None


class LinkedList:  # Creating a linked list
    def __init__(self):
        self.head = None

    def print_list(self) -> None:
        cur_node: Optional[Node] = self.head
        while cur_node:
            print(cur_node.item)
            cur_node = cur_node.next

    def append(self, item: Any) -> None:
        new_node: Node = Node(item)
        if self.head is None:
            self.head = new_node
            return
        last_node: Optional[Node] = self.head
        while last_node.next:
            last_node: Optional[Node] = last_node.next
        last_node.next = new_node

    def prepend(self, item: Any) -> None:
        new_node: Node = Node(item)
        new_node.next = self.head
        self.head = new_node

    def insert_after_node(self, prev_node: Optional[Node], item: Any) -> None:
        if not prev_node:
            print("Previous node is not in the list")
            return
        new_node: Node = Node(item)
        new_node.next = prev_node.next
        prev_node.next = new_node

    def delete_node(self, key: Any) -> None:
        cur_node: Optional[Node] = self.head
        if cur_node and cur_node.item == key:
            self.head = cur_node.next
            cur_node = None
            return
        prev: Optional[Node] = None
        while cur_node and cur_node.item != key:
            prev = cur_node
            cur_node: Optional[Node] = cur_node.next
        if cur_node is None:
            return
        prev.next = cur_node.next
        cur_node = None

    def delete_node_at_pos(self, pos: int) -> None:
        cur_node: Optional[Node] = self.head
        if pos == 0:
            self.head = cur_node.next
            cur_node = None
            return
        prev: Optional[Node] = None
        count: int = 1
        while cur_node and count != pos:
            prev = cur_node
            cur_node: Optional[Node] = cur_node.next
            count += 1
        if cur_node is None:
            return
        prev.next = cur_node.next
        cur_node = None

    def len_iterative(self) -> int:
        count: int = 0
        cur_node: Optional[Node] = self.head
        while cur_node:
            count += 1
            cur_node = cur_node.next
        return count

    def len_recursive(self, node: Optional[Node]) -> int:
        if node is None:
            return 0
        return 1 + self.len_recursive(node.next)

    def swap_nodes(self, key1: Any, key2: Any) -> None:
        if key1 == key2:
            return
        prev1: Optional[Node] = None
        cur1: Optional[Node] = self.head
        while cur1 and cur1.item != key1:
            prev1 = cur1
            cur1 = cur1.next
        prev2: Optional[Node] = None
        cur2: Optional[Node] = self.head
        while cur2 and cur2.item != key2:
            prev2 = cur2
            cur2 = cur2.next
        if not cur1 or not cur2:
            return
        if prev1:
            prev1.next = cur2
        else:
            self.head = cur2
        if prev2:
            prev2.next = cur1
        else:
            self.head = cur1
        cur1.next, cur2.next = cur2.next, cur1.next

    def reverse_iterative(self) -> None:
        prev: Optional[Node] = None
        cur: Optional[Node] = self.head
        while cur:
            nxt: Optional[Node] = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        self.head = prev

    def reverse_recursive(self) -> None:
        def _reverse_recursive(
            cur: Optional[Node], prev: Optional[Node]
        ) -> Optional[Node]:
            if not cur:
                return prev
            nxt: Optional[Node] = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
            return _reverse_recursive(cur, prev)

        self.head = _reverse_recursive(cur=self.head, prev=None)

    def merge_sorted(self, llist: "LinkedList") -> None:
        p: Optional[Node] = self.head
        q: Optional[Node] = llist.head
        s: Optional[Node] = None
        new_head: Optional[Node] = None
        if not p:
            return q
        if not q:
            return p
        if p and q:
            if p.item <= q.item:
                s = p
                p = s.next
            else:
                s = q
                q = s.next
            new_head = s
        while p and q:
            if p.item <= q.item:
                s.next = p
                s = p
                p = s.next
            else:
                s.next = q
                s = q
                q = s.next
        if not p:
            s.next = q
        if not q:
            s.next = p
        return new_head

    def remove_duplicates(self):
        cur: Optional[Node] = self.head
        prev: Optional[Node] = None
        dup_values: Dict[Any, int] = {}
        while cur:
            if cur.item in dup_values:
                prev.next = cur.next
                cur = None
            else:
                dup_values[cur.item] = 1
                prev = cur
            cur = prev.next

    def print_nth_from_last(self, n):
        total_len: int = self.len_iterative()
        cur: Optional[Node] = self.head
        while cur:
            if total_len == n:
                print(cur.item)
                return cur.item
            total_len -= 1
            cur = cur.next
        if cur is None:
            return

    def rotate(self, k):
        p: Optional[Node] = self.head
        q: Optional[Node] = self.head
        prev: Optional[Node] = None
        count: int = 0
        while p and count < k:
            prev = p
            p = p.next
            q = q.next
            count += 1
        p = prev
        while q:
            prev = q
            q = q.next
        q = prev
        q.next = self.head
        self.head = p.next
        p.next = None

    def is_palindrome(self) -> bool:
        s: str = ""
        p: Optional[Node] = self.head
        while p:
            s += p.item
            p = p.next
        return s == s[::-1]

    def move_tail_to_head(self):
        last: Optional[Node] = self.head
        second_to_last: Optional[Node] = None
        while last.next:
            second_to_last = last
            last = last.next
        last.next = self.head
        second_to_last.next = None
        self.head = last

    def sum_two_lists(self, llist: "LinkedList") -> "LinkedList":
        p: Optional[Node] = self.head
        q: Optional[Node] = llist.head
        sum_llist: LinkedList = LinkedList()
        carry: int = 0
        while p or q:
            if not p:
                i: int = 0
            else:
                i: int = p.item
            if not q:
                j = 0
            else:
                j: int = q.item
            s: int = i + j + carry
            if s >= 10:
                carry = 1
                remainder: int = s % 10
                sum_llist.append(remainder)
            else:
                carry = 0
                sum_llist.append(s)
            if p:
                p = p.next
            if q:
                q = q.next
        return sum_llist

    def is_circular_linked_list(self, input_list: "LinkedList") -> bool:
        cur: Optional[Node] = input_list.head
        while cur:
            cur = cur.next
            if cur == input_list.head:
                return True
        return False

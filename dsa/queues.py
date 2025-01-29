from typing import List, Any, Optional


class Queue:  # Creating a queue
    def __init__(self) -> None:
        self.queue: List[Any] = []

    def enqueue(self, item: Any) -> None:  # Add an element
        self.queue.append(item)

    def dequeue(self) -> Optional[Any]:  # Remove an element
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def display(self) -> None:  # Display  the queue
        print(self.queue)

    def size(self) -> int:
        return len(self.queue)


class PriorityQueue(Queue):
    @staticmethod
    def heapify(arr: List[Any], n: int, i: int) -> None:  # Function to heapify the tree
        # Find the largest among root, left child and right child
        largest: int = i
        l: int = 2 * i + 1
        r: int = 2 * i + 2
        if l < n and arr[i] < arr[l]:
            largest = l
        if r < n and arr[largest] < arr[r]:
            largest = r
        # Swap and continue heapifying if root is not largest
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            PriorityQueue.heapify(arr, n, largest)

    @staticmethod
    def insert(
        array: List[Any], newNum: int
    ) -> None:  # Function to insert an element into the tree
        size: int = len(array)
        if size == 0:
            array.append(newNum)
        else:
            array.append(newNum)
            for i in range((size // 2) - 1, -1, -1):
                PriorityQueue.heapify(array, size, i)

    @staticmethod
    def deleteNode(
        array: List[Any], num: int
    ) -> None:  # Function to delete an element from the tree
        size: int = len(array)
        i: int = 0
        for i in range(0, size):
            if num == array[i]:
                break
        array[i], array[size - 1] = array[size - 1], array[i]
        array.remove(size - 1)

        for i in range((len(array) // 2) - 1, -1, -1):
            PriorityQueue.heapify(array, len(array), i)


class Deque:  # Creating deque class
    def __init__(self) -> None:  # Initializing deque
        self.items: List[Any] = []

    def isEmpty(self):  # Check deque is empty
        return self.items == []

    def addRear(self, item: Any) -> None:  # Add an element from rear
        self.items.append(item)

    def addFront(self, item: Any) -> None:  # Add an element from front
        self.items.insert(0, item)

    def removeFront(self) -> Optional[Any]:  # Remove an element from front
        return self.items.pop(0)

    def removeRear(self) -> Optional[Any]:  # Remove an element from rear
        return self.items.pop()

    def size(self) -> int:  # Get the size of deque
        return len(self.items)


class CircularQueue(object):
    def __init__(self, k: int) -> None:
        self.k: int = k
        self.queue: List[Optional[Any]] = [None] * k
        self.head = self.tail = -1

    def enqueue(self, data: Any) -> None:  # Insert an element into the circular queue
        if (self.tail + 1) % self.k == self.head:
            print("The circular queue is full\n")
        elif self.head == -1:
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data
        else:
            self.tail = (self.tail + 1) % self.k
            self.queue[self.tail] = data

    def dequeue(self) -> Optional[Any]:  # Delete an element from the circular queue
        if self.head == -1:
            print("The circular queue is empty\n")
        elif self.head == self.tail:
            temp = self.queue[self.head]
            self.head = -1
            self.tail = -1
            return temp
        else:
            temp = self.queue[self.head]
            self.head = (self.head + 1) % self.k
            return temp

    def printCQueue(self):
        if self.head == -1:
            print("No element in the circular queue")
        elif self.tail >= self.head:
            for i in range(self.head, self.tail + 1):
                print(self.queue[i], end=" ")
            print()
        else:
            for i in range(self.head, self.k):
                print(self.queue[i], end=" ")
            for i in range(0, self.tail + 1):
                print(self.queue[i], end=" ")
            print()

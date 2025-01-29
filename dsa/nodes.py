from typing import Optional, List
from utils.helpers import Height


class Node:
    def __init__(self, item: int) -> None:
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.val: int = item

    @staticmethod
    def inorder(root: Optional["Node"]) -> None:
        if root:
            Node.inorder(root.left)  # Traverse left
            print(str(root.val) + "->", end="")  # Traverse root
            Node.inorder(root.right)  # Traverse right

    @staticmethod
    def postorder(root: Optional["Node"]) -> None:
        if root:
            Node.postorder(root.left)  # Traverse left
            Node.postorder(root.right)  # Traverse right
            print(str(root.val) + "->", end="")  # Traverse root

    @staticmethod
    def preorder(root: Optional["Node"]) -> None:
        if root:
            print(str(root.val) + "->", end="")  # Traverse root
            Node.preorder(root.left)  # Traverse left
            Node.preorder(root.right)  # Traverse right

    @staticmethod
    def isFullTree(root: Optional["Node"]) -> bool:  # Checking full binary tree
        if root is None:  # Tree empty case
            return True
        if (
            root.left is None and root.rightChild is None
        ):  # Checking whether child is present
            return True
        if root.left is not None and root.rightChild is not None:
            return Node.isFullTree(root.left) and Node.isFullTree(root.rightChild)
        return False

    @staticmethod
    def calculateDepth(node: Optional["Node"]) -> int:  # Calculate the depth
        d: int = 0
        while node is not None:
            d += 1
            node = node.left
        return d

    @staticmethod
    def is_perfect(
        root: Optional["Node"], d: int, level: int = 0
    ) -> bool:  # Check if the tree is perfect binary tree
        if root is None:  # Check if the tree is empty
            return True
        if root.left is None and root.right is None:  # Check the presence of trees
            return d == level + 1
        if root.left is None or root.right is None:
            return False
        return Node.is_perfect(root.left, d, level + 1) and Node.is_perfect(
            root.right, d, level + 1
        )

    @staticmethod
    def count_nodes(root: Optional["Node"]) -> int:  # Count the number of nodes
        if root is None:
            return 0
        return 1 + Node.count_nodes(root.left) + Node.count_nodes(root.right)

    @staticmethod
    def is_complete(
        root: Optional["Node"], index: int, numberNodes: int
    ) -> bool:  # Check if the tree is complete binary tree
        if root is None:  # Check if the tree is empty
            return True
        if index >= numberNodes:
            return False
        return Node.is_complete(
            root.left, 2 * index + 1, numberNodes
        ) and Node.is_complete(root.right, 2 * index + 2, numberNodes)

    @staticmethod
    def insert(node: Optional["Node"], key: int) -> Optional["Node"]:  # Insert a node
        if node is None:  # Return a new node if the tree is empty
            return Node(key)
        if key < node.key:  # Traverse to the right place and insert the node
            node.left = Node.insert(node.left, key)
        else:
            node.right = Node.insert(node.right, key)
        return node

    @staticmethod
    def minValueNode(
        node: Optional["Node"],
    ) -> Optional["Node"]:  # Find the inorder successor
        current: Optional[Node] = node
        while current.left is not None:  # Find the leftmost leaf
            current = current.left
        return current

    @staticmethod
    def deleteNode(
        root: Optional["Node"], key: int
    ) -> Optional["Node"]:  # Deleting a node
        if root is None:  # Return if the tree is empty
            return root
        if key < root.key:  # Find the node to be deleted
            root.left = Node.deleteNode(root.left, key)
        elif key > root.key:
            root.right = Node.deleteNode(root.right, key)
        else:
            if root.left is None:  # If the node is with only one child or no child
                temp: Node = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            temp: Node = Node.minValueNode(
                root.right
            )  # If the node has two children, place the inorder successor in position of the node to be deleted

            root.key = temp.key

            root.right = Node.deleteNode(
                root.right, temp.key
            )  # Delete the inorder successor

        return root

    @staticmethod
    def isHeightBalanced(
        root, height
    ) -> bool:  # Checking if the tree is height balanced
        left_height: Height = Height()
        right_height: Height = Height()

        if root is None:
            return True

        l: bool = Node.isHeightBalanced(root.left, left_height)
        r: bool = Node.isHeightBalanced(root.right, right_height)

        height.height = max(left_height.height, right_height.height) + 1

        if abs(left_height.height - right_height.height) <= 1:
            return l and r

        return False


class SinglyLinkedListNode:

    def __init__(
        self,
        value: int = 0,
        # pylint: disable=W0622
        next: Optional["SinglyLinkedListNode"] = None,
    ) -> None:
        self.value: int = value
        self.next: Optional[SinglyLinkedListNode] = next

    def length(self) -> int:
        head: Optional[SinglyLinkedListNode] = self
        count: int = 0
        while head:
            count += 1
            head = head.next
        return count

    def __str__(self) -> str:
        res: List[str] = []
        current: Optional[SinglyLinkedListNode] = self
        while current:
            res.append(str(current.value))
            current = current.next
        return " -> ".join(res)


class TreeNode(object):  # Create a tree node
    def __init__(self, key: int) -> None:
        self.key: int = key
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.height: int = 1


class RBNode:  # Node creation
    def __init__(self, item: int) -> None:
        self.item: int = item
        self.parent: Optional[RBNode] = None
        self.left: Optional[RBNode] = None
        self.right: Optional[RBNode] = None
        self.color: int = 1


class BPlusNode:  # Node creation
    def __init__(self, leaf: bool = False) -> None:
        self.leaf: bool = leaf
        self.keys: List[int] = []
        self.child: List[BPlusNode] = []


class AdjNode:  # Adjacency list node
    def __init__(self, value: int) -> None:
        self.vertex: int = value
        self.next: Optional[AdjNode] = None

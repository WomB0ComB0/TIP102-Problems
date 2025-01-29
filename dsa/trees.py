from .nodes import TreeNode, BPlusNode, RBNode
import sys
import math
from typing import List, Optional


class AVLTree(object):  # Create an avl tree class with insert and delete functions
    def insert_node(
        self, root: Optional[TreeNode], key: int
    ) -> TreeNode:  # Function to insert a node
        if root is None:  # Find the correct location and insert the node
            return TreeNode(key)
        elif key < root.key:
            root.left = self.insert_node(root.left, key)
        else:
            root.right = self.insert_node(root.right, key)
        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))

        # Update the balance factor and balance the tree
        balanceFactor: int = self.getBalance(root)
        if balanceFactor > 1:
            if key < root.left.key:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balanceFactor < -1:
            if key > root.right.key:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        return root

    def delete_node(
        self, root: Optional[TreeNode], key: int
    ) -> Optional[TreeNode]:  # Function to delete a node
        if root is None:  # Find the node to be deleted and remove it
            return root
        elif key < root.key:
            root.left = self.delete_node(root.left, key)
        elif key > root.key:
            root.right = self.delete_node(root.right, key)
        else:
            if root.left is None:
                temp: Optional[TreeNode] = root.right
                root = None
                return temp
            elif root.right is None:
                temp: Optional[TreeNode] = root.left
                root = None
                return temp
            temp: Optional[TreeNode] = self.getMinValueNode(root.right)
            root.key = temp.key
            root.right = self.delete_node(root.right, temp.key)
        if root is None:
            return root
        root.height = 1 + max(
            self.getHeight(root.left), self.getHeight(root.right)
        )  # Update the balance factor of nodes
        balanceFactor: int = self.getBalance(root)
        if balanceFactor > 1:  # Balance the tree
            if self.getBalance(root.left) >= 0:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balanceFactor < -1:
            if self.getBalance(root.right) <= 0:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        return root

    def leftRotate(self, z: TreeNode) -> TreeNode:  # Function to perform left rotation
        y: TreeNode = z.right
        T2: Optional[TreeNode] = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))
        return y

    def rightRotate(
        self, z: TreeNode
    ) -> TreeNode:  # Function to perform right rotation
        y: TreeNode = z.left
        T3: Optional[TreeNode] = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))
        return y

    def getHeight(self, root: Optional[TreeNode]) -> int:  # Get the height of the node
        if not root:
            return 0
        return root.height

    def getBalance(
        self, root: Optional[TreeNode]
    ) -> int:  # Get balance factor of the node
        if not root:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def getMinValueNode(
        self, root: Optional[TreeNode]
    ) -> Optional[TreeNode]:  # Get the node with minimum value
        if root is None or root.left is None:
            return root
        return self.getMinValueNode(root.left)

    def preOrder(self, root: Optional[TreeNode]) -> None:  # Preorder tree traversal
        if not root:
            return
        print("{0} ".format(root.key), end="")
        self.preOrder(root.left)
        self.preOrder(root.right)

    def printHelper(
        self, currPtr: Optional[TreeNode], indent: str, last: bool
    ) -> None:  # Print the tree
        if currPtr is not None:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "
            print(currPtr.key)
            self.printHelper(currPtr.left, indent, False)
            self.printHelper(currPtr.right, indent, True)


class RedBlackTree:  # Red-Black Tree
    def __init__(self) -> None:
        self.TNULL: RBNode = RBNode(None)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root: RBNode = self.TNULL

    def pre_order_helper(self, node: RBNode) -> None:  # Preorder
        if node != self.TNULL:
            sys.stdout.write(str(node.item) + " ")
            self.pre_order_helper(node.left)
            self.pre_order_helper(node.right)

    def in_order_helper(self, node: RBNode) -> None:  # Inorder
        if node != self.TNULL:
            self.in_order_helper(node.left)
            sys.stdout.write(str(node.item) + " ")
            self.in_order_helper(node.right)

    def post_order_helper(self, node: RBNode) -> None:  # Postorder
        if node != self.TNULL:
            self.post_order_helper(node.left)
            self.post_order_helper(node.right)
            sys.stdout.write(str(node.item) + " ")

    def search_tree_helper(self, node: RBNode, key: int) -> RBNode:  # Search the tree
        if node == self.TNULL or key == node.item:
            return node
        if key < node.item:
            return self.search_tree_helper(node.left, key)
        return self.search_tree_helper(node.right, key)

    def delete_fix(
        self, x: Optional[RBNode]
    ) -> None:  # Balancing the tree after deletion
        while x is not None and x.color == 0:
            if x == x.parent.left:
                s: RBNode = x.parent.right
                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    s = x.parent.right
                if s.left.color == 0 and s.right.color == 0:
                    s.color = 1
                    x = x.parent
                else:
                    if s.right.color == 0:
                        s.left.color = 0
                        s.color = 1
                        self.right_rotate(s)
                        s = x.parent.right
                    s.color = x.parent.color
                    x.parent.color = 0
                    s.right.color = 0
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    s = x.parent.left
                if s.right.color == 0 and s.left.color == 0:
                    s.color = 1
                    x = x.parent
                else:
                    if s.left.color == 0:
                        s.right.color = 0
                        s.color = 1
                        self.left_rotate(s)
                        s = x.parent.left
                    s.color = x.parent.color
                    x.parent.color = 0
                    s.left.color = 0
                    self.right_rotate(x.parent)
                    x = self.root
        if x is not None:
            x.color = 0

    def __rb_transplant(self, u: Optional[RBNode], v: Optional[RBNode]) -> None:
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def delete_node_helper(
        self, node: Optional[RBNode], key: int
    ) -> None:  # Node deletion
        z: Optional[RBNode] = self.TNULL
        while node is not None:
            if node.item == key:
                z = node
            if node.item <= key:
                node = node.right
            else:
                node = node.left
        if z == self.TNULL:
            print("Cannot find key in the tree")
            return
        y: Optional[RBNode] = z
        y_original_color: int = y.color
        if z.left == self.TNULL:
            x: Optional[RBNode] = z.right
            self.__rb_transplant(z, z.right)
        elif z.right == self.TNULL:
            x: Optional[RBNode] = z.left
            self.__rb_transplant(z, z.left)
        else:
            y: Optional[RBNode] = self.minimum(z.right)
            y_original_color: int = y.color
            x: Optional[RBNode] = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.right)
                y.right = z.right
                if y.right is not None:
                    y.right.parent = y
            self.__rb_transplant(z, y)
            y.left = z.left
            if y.left is not None:
                y.left.parent = y
            y.color = z.color
        if y_original_color == 0:
            self.delete_fix(x)

    def fix_insert(
        self, k: Optional[RBNode]
    ) -> None:  # Balance the tree after insertion
        while k is not None and k.parent is not None and k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u: Optional[RBNode] = k.parent.parent.left
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u: Optional[RBNode] = k.parent.parent.right
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    def __print_helper(
        self, node: Optional[RBNode], indent: str, last: bool
    ) -> None:  # Printing the tree
        if node is not None:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "
            s_color: str = "RED" if node.color == 1 else "BLACK"
            print(str(node.item) + "(" + s_color + ")")
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

    def preorder(self) -> None:
        if self.root is not None:
            self.pre_order_helper(self.root)

    def inorder(self) -> None:
        if self.root is not None:
            self.in_order_helper(self.root)

    def postorder(self) -> None:
        if self.root is not None:
            self.post_order_helper(self.root)

    def searchTree(self, k: int) -> Optional[RBNode]:
        if self.root is not None:
            return self.search_tree_helper(self.root, k)
        return None

    def minimum(self, node: Optional[RBNode]) -> Optional[RBNode]:
        while node is not None and node.left is not self.TNULL:
            node = node.left
        return node

    def maximum(self, node: Optional[RBNode]) -> Optional[RBNode]:
        while node is not None and node.right is not self.TNULL:
            node = node.right
        return node

    def successor(self, x: Optional[RBNode]) -> Optional[RBNode]:
        if x is not None and x.right is not self.TNULL:
            return self.minimum(x.right)
        y: Optional[RBNode] = x.parent
        while y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    def predecessor(
        self, x: Optional[RBNode]
    ) -> Optional[RBNode]:  # x is the node to be deleted
        if x is not None and x.left is not self.TNULL:
            return self.maximum(x.left)
        y: Optional[RBNode] = x.parent
        while y is not None and x == y.left:
            x = y
            y = y.parent
        return y

    def left_rotate(self, x: Optional[RBNode]) -> None:  # x is the node to be rotated
        if x is not None and x.right is not self.TNULL:
            y: Optional[RBNode] = x.right
            x.right = y.left
            if y.left is not None:
                y.left.parent = x
            y.parent = x.parent
            if x.parent is None:
                self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x: Optional[RBNode]) -> None:  # x is the node to be rotated
        if x is not None and x.left is not self.TNULL:
            y: Optional[RBNode] = x.left
            x.left = y.right
            if y.right is not None:
                y.right.parent = x
            y.parent = x.parent
            if x.parent is None:
                self.root = y
            elif x == x.parent.right:
                x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, key: int) -> None:
        node: RBNode = RBNode(key)
        node.parent = None
        node.item = key
        node.left = self.TNULL
        node.right = self.TNULL
        node.color = 1

        y: Optional[RBNode] = None
        x: Optional[RBNode] = self.root

        while x != self.TNULL:
            y = x
            if node.item < x.item:
                x = x.left
            else:
                x = x.right
        node.parent = y
        if y is None:
            self.root = node
        elif node.item < y.item:
            y.left = node
        else:
            y.right = node
        if node.parent is None:
            node.color = 0
            return
        if node.parent.parent is None:
            return

        self.fix_insert(node)

    def get_root(self):
        return self.root

    def delete_node(self, item: int) -> None:
        self.delete_node_helper(self.root, item)

    def print_tree(self):
        self.__print_helper(self.root, "", True)


# B plus tree
class BplusTree:
    def __init__(self, t):
        self.root = BPlusNode(True)
        self.t = t

    def insert(self, k: int) -> None:  # Insert a key
        root: Optional[BPlusNode] = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp: Optional[BPlusNode] = BPlusNode()
            self.root = temp
            temp.child.insert(0, root)
            self.split_child(temp, 0)
            self.insert_non_full(temp, k)
        else:
            self.insert_non_full(root, k)

    def insert_non_full(
        self, x: Optional[BPlusNode], k: int
    ) -> None:  # Insert non full
        i: int = len(x.keys) - 1
        if x.leaf:
            x.keys.append((None, None))
            while i >= 0 and k[0] < x.keys[i][0]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k[0] < x.keys[i][0]:
                i -= 1
            i += 1
            if len(x.child[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                if k[0] > x.keys[i][0]:
                    i += 1
            self.insert_non_full(x.child[i], k)

    def split_child(self, x, i):  # Split the child
        t: int = self.t
        y: Optional[BPlusNode] = x.child[i]
        z: Optional[BPlusNode] = BPlusNode(y.leaf)
        x.child.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        z.keys = y.keys[t : (2 * t) - 1]
        y.keys = y.keys[0 : t - 1]
        if not y.leaf:
            z.child = y.child[t : 2 * t]
            y.child = y.child[0 : t - 1]

    def search(self, value):  # Search operation for different operations
        current_node: Optional[BPlusNode] = self.root
        while current_node is not None and current_node.check_leaf is False:
            temp2: List[int] = current_node.values
            for i, item in enumerate(temp2):
                if value == item:
                    current_node = current_node.keys[i + 1]
                    break
                elif value < item:
                    current_node = current_node.keys[i]
                    break
                elif i + 1 == len(current_node.values):
                    current_node = current_node.keys[i + 1]
                    break
        return current_node

    def find(self, value: int, key: int) -> bool:  # Find the node
        l: Optional[BPlusNode] = self.search(value)
        for i, item in enumerate(l.values):
            if item == value:
                if key in l.keys[i]:
                    return True
                else:
                    return False
        return False

    def insert_in_parent(
        self, n: BPlusNode, value: int, ndash: BPlusNode
    ) -> None:  # Inserting at the parent
        if self.root == n:
            rootNode: Optional[BPlusNode] = BPlusNode(n.order)
            rootNode.values = [value]
            rootNode.keys = [n, ndash]
            self.root = rootNode
            n.parent = rootNode
            ndash.parent = rootNode
            return
        parentNode: Optional[BPlusNode] = n.parent
        temp3: List[Optional[BPlusNode]] = parentNode.keys
        for i, item in enumerate(temp3):
            if item == n:
                parentNode.values = (
                    parentNode.values[:i] + [value] + parentNode.values[i:]
                )
                parentNode.keys = (
                    parentNode.keys[: i + 1] + [ndash] + parentNode.keys[i + 1 :]
                )
                if len(parentNode.keys) > parentNode.order:
                    parentdash: Optional[BPlusNode] = BPlusNode(parentNode.order)
                    parentdash.parent = parentNode.parent
                    mid: int = int(math.ceil(parentNode.order / 2)) - 1
                    parentdash.values = parentNode.values[mid + 1 :]
                    parentdash.keys = parentNode.keys[mid + 1 :]
                    value_: int = parentNode.values[mid]
                    if mid == 0:
                        parentNode.values = parentNode.values[: mid + 1]
                    else:
                        parentNode.values = parentNode.values[:mid]
                    parentNode.keys = parentNode.keys[: mid + 1]
                    for j in parentNode.keys:
                        j.parent = parentNode
                    for j in parentdash.keys:
                        j.parent = parentdash
                    self.insert_in_parent(parentNode, value_, parentdash)

    def delete(self, x: Optional[BPlusNode], k: int) -> None:  # Delete a node
        t: int = self.t
        i: int = 0
        while i < len(x.keys) and k[0] > x.keys[i][0]:
            i += 1
        if x.leaf:
            if i < len(x.keys) and x.keys[i][0] == k[0]:
                x.keys.pop(i)
                return
            return
        if i < len(x.keys) and x.keys[i][0] == k[0]:
            return self.delete_internal_node(x, k, i)
        elif len(x.child[i].keys) >= t:
            self.delete(x.child[i], k)
        else:
            if i != 0 and i + 2 < len(x.child):
                if len(x.child[i - 1].keys) >= t:
                    self.delete_sibling(x, i, i - 1)
                elif len(x.child[i + 1].keys) >= t:
                    self.delete_sibling(x, i, i + 1)
                else:
                    self.delete_merge(x, i, i + 1)
            elif i == 0:
                if len(x.child[i + 1].keys) >= t:
                    self.delete_sibling(x, i, i + 1)
                else:
                    self.delete_merge(x, i, i + 1)
            elif i + 1 == len(x.child):
                if len(x.child[i - 1].keys) >= t:
                    self.delete_sibling(x, i, i - 1)
                else:
                    self.delete_merge(x, i, i - 1)
            self.delete(x.child[i], k)

    def delete_internal_node(
        self, x: Optional[BPlusNode], k: int, i: int
    ) -> None:  # Delete internal node
        t: int = self.t
        if x is not None and x.leaf:
            if x.keys[i][0] == k[0]:
                x.keys.pop(i)
                return
            return
        if len(x.child[i].keys) >= t:
            x.keys[i] = self.delete_predecessor(x.child[i])
            return
        elif len(x.child[i + 1].keys) >= t:
            x.keys[i] = self.delete_successor(x.child[i + 1])
            return
        else:
            self.delete_merge(x, i, i + 1)
            self.delete_internal_node(x.child[i], k, self.t - 1)

    def delete_predecessor(
        self, x: Optional[BPlusNode]
    ) -> int:  # Delete the predecessor
        if x is not None and x.leaf:
            return x.pop()
        n: int = len(x.keys) - 1
        if len(x.child[n].keys) >= self.t:
            self.delete_sibling(x, n + 1, n)
        else:
            self.delete_merge(x, n, n + 1)
        self.delete_predecessor(x.child[n])

    def delete_successor(self, x: Optional[BPlusNode]) -> int:  # Delete the successor
        if x is not None and x.leaf:
            return x.keys.pop(0)
        if len(x.child[1].keys) >= self.t:
            self.delete_sibling(x, 0, 1)
        else:
            self.delete_merge(x, 0, 1)
        self.delete_successor(x.child[0])

    def delete_merge(
        self, x: Optional[BPlusNode], i: int, j: int
    ) -> None:  # Delete resolution
        cnode: Optional[BPlusNode] = x.child[i]
        if j > i:
            rsnode: Optional[BPlusNode] = x.child[j]
            cnode.keys.append(x.keys[i])
            for k, _ in enumerate(rsnode.keys):
                cnode.keys.append(rsnode.keys[k])
                if len(rsnode.child) > 0:
                    cnode.child.append(rsnode.child[k])
            if len(rsnode.child) > 0:
                cnode.child.append(rsnode.child.pop())
            new: Optional[BPlusNode] = cnode
            x.keys.pop(i)
            x.child.pop(j)
        else:
            lsnode: Optional[BPlusNode] = x.child[j]
            lsnode.keys.append(x.keys[j])
            for i, _ in enumerate(cnode.keys):
                lsnode.keys.append(cnode.keys[i])
                if len(lsnode.child) > 0:
                    lsnode.child.append(cnode.child[i])
            if len(lsnode.child) > 0:
                lsnode.child.append(cnode.child.pop())
            new = lsnode
            x.keys.pop(j)
            x.child.pop(i)
        if x == self.root and len(x.keys) == 0:
            self.root = new

    def delete_sibling(
        self, x: Optional[BPlusNode], i: int, j: int
    ) -> None:  # Delete the sibling
        cnode: Optional[BPlusNode] = x.child[i]
        if i < j:
            rsnode: Optional[BPlusNode] = x.child[j]
            cnode.keys.append(x.keys[i])
            x.keys[i] = rsnode.keys[0]
            if len(rsnode.child) > 0:
                cnode.child.append(rsnode.child[0])
                rsnode.child.pop(0)
            rsnode.keys.pop(0)
        else:
            lsnode: Optional[BPlusNode] = x.child[j]
            cnode.keys.insert(0, x.keys[i - 1])
            x.keys[i - 1] = lsnode.keys.pop()
            if len(lsnode.child) > 0:
                cnode.child.insert(0, lsnode.child.pop())

    def deleteEntry(
        self, node_: Optional[BPlusNode], value: int, key: int
    ) -> None:  # Delete an entry
        if node_ is not None and not node_.check_leaf:
            for i, item in enumerate(node_.keys):
                if item == key:
                    node_.keys.pop(i)
                    break
            for i, item in enumerate(node_.values):
                if item == value:
                    node_.values.pop(i)
                    break
        if self.root == node_ and len(node_.keys) == 1:
            self.root = node_.keys[0]
            node_.keys[0].parent = None
            del node_
            return
        elif (
            len(node_.keys) < int(math.ceil(node_.order / 2))
            and node_.check_leaf is False
        ) or (
            len(node_.values) < int(math.ceil((node_.order - 1) / 2))
            and node_.check_leaf is True
        ):
            is_predecessor = 0
            parentNode: Optional[BPlusNode] = node_.parent
            PrevNode: int = -1
            NextNode: int = -1
            PrevK: int = -1
            PostK: int = -1
            for i, item in enumerate(parentNode.keys):
                if item == node_:
                    if i > 0:
                        PrevNode = parentNode.keys[i - 1]
                        PrevK = parentNode.values[i - 1]
                    if i < len(parentNode.keys) - 1:
                        NextNode = parentNode.keys[i + 1]
                        PostK = parentNode.values[i]
            if PrevNode == -1:
                ndash: Optional[BPlusNode] = NextNode
                value_: int = PostK
            elif NextNode == -1:
                is_predecessor = 1
                ndash: Optional[BPlusNode] = PrevNode
                value_: int = PrevK
            else:
                if len(node_.values) + len(NextNode.values) < node_.order:
                    ndash: Optional[BPlusNode] = NextNode
                    value_: int = PostK
                else:
                    is_predecessor = 1
                    ndash: Optional[BPlusNode] = PrevNode
                    value_: int = PrevK
            if len(node_.values) + len(ndash.values) < node_.order:
                if is_predecessor == 0:
                    node_, ndash = ndash, node_
                ndash.keys += node_.keys
                if node_ is not None and not node_.check_leaf:
                    ndash.values.append(value_)
                else:
                    ndash.nextKey = node_.nextKey
                ndash.values += node_.values
                if ndash is not None and not ndash.check_leaf:
                    for j in ndash.keys:
                        j.parent = ndash
                self.deleteEntry(node_.parent, value_, node_)
                del node_
            else:
                if is_predecessor == 1:
                    if node_ is not None and not node_.check_leaf:
                        ndashpm: int = ndash.keys.pop(-1)
                        ndashkm_1: int = ndash.values.pop(-1)
                        node_.keys = [ndashpm] + node_.keys
                        node_.values = [value_] + node_.values
                        parentNode = node_.parent
                        for i, item in enumerate(parentNode.values):
                            if item == value_:
                                parentNode.values[i] = ndashkm_1
                                break
                    else:
                        ndashpm: int = ndash.keys.pop(-1)
                        ndashkm: int = ndash.values.pop(-1)
                        node_.keys = [ndashpm] + node_.keys
                        node_.values = [ndashkm] + node_.values
                        parentNode = node_.parent
                        for i, item in enumerate(parentNode.values):
                            if item == value_:
                                parentNode.values[i] = ndashkm
                                break
                else:
                    if not node_.check_leaf:
                        ndashp0: int = ndash.keys.pop(0)
                        ndashk0: int = ndash.values.pop(0)
                        node_.keys = node_.keys + [ndashp0]
                        node_.values = node_.values + [value_]
                        parentNode = node_.parent
                        for i, item in enumerate(parentNode.values):
                            if item == value_:
                                parentNode.values[i] = ndashk0
                                break
                    else:
                        ndashp0: int = ndash.keys.pop(0)
                        ndashk0: int = ndash.values.pop(0)
                        node_.keys = node_.keys + [ndashp0]
                        node_.values = node_.values + [ndashk0]
                        parentNode = node_.parent
                        for i, item in enumerate(parentNode.values):
                            if item == value_:
                                parentNode.values[i] = ndash.values[0]
                                break
                if ndash is not None and not ndash.check_leaf:
                    for j in ndash.keys:
                        j.parent = ndash
                if node_ is not None and not node_.check_leaf:
                    for j in node_.keys:
                        j.parent = node_
                if parentNode is not None and not parentNode.check_leaf:
                    for j in parentNode.keys:
                        j.parent = parentNode

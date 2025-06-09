class Node:
    """A node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """A singly linked list."""
    def __init__(self):
        self.head = None

    def add(self, data):
        """Add a node with the given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        """Print the contents of the list."""
        if not self.head:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        """Delete the nth node (1-based index)."""
        if n <= 0:
            raise ValueError("Index should be 1 or greater.")

        if not self.head:
            raise IndexError("Cannot delete from an empty list.")

        if n == 1:
            self.head = self.head.next
            return

        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1

        if not current or not current.next:
            raise IndexError("Index out of range.")

        current.next = current.next.next


# Testing the implementation
if __name__ == "__main__":
    ll = LinkedList()
    # Add elements
    ll.add(10)
    ll.add(20)
    ll.add(30)
    ll.add(40)
    print("Original list:")
    ll.print_list()

    # Delete 2nd node
    ll.delete_nth_node(2)
    print("\nList after deleting 2nd node:")
    ll.print_list()

    # Attempt to delete node with invalid index
    try:
        ll.delete_nth_node(10)
    except Exception as e:
        print(f"\nException: {e}")

    # Delete head
    ll.delete_nth_node(1)
    print("\nList after deleting head (1st node):")
    ll.print_list()

    # Empty the list and try deletion
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)
    try:
        ll.delete_nth_node(1)
    except Exception as e:
        print(f"\nException when deleting from empty list: {e}")

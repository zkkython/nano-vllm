from collections import deque


q = deque()

q.extendleft([1, 2, 3])


while q:
    print(q.popleft())

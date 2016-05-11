
# coding: utf-8

import inspect

class abstractmethod(object):

    def __init__(self, func):
        assert inspect.isfunction(func) 
        self._func = func

    def __get__(self, obj, type):
        return self.method(obj, self._func, type)

    class method(object):

        def __init__(self, obj, func, cls):
            self._self = obj
            self._func = func
            self._class = cls
            self.__name__ = func.__name__

        def __call__(self, *args, **kwargs):
            msg = "Abstract method %s of class %s called." % (
                    self._func.__name__, self._class.__name__)
            raise TypeError("msg")
            
class Object(object):

    def __init__(self):
        super(Object, self).__init__()

    def __cmp__(self, obj):
        if isinstance(self, obj.__class__):
            return self._compareTo(obj)
        elif isinstance(obj, self.__class__):
            return -obj._compareTo(self)
        else:
            return cmp(self.__class__.__name__,
                obj.__class__.__name__)

    def _compareTo(self, obj):
        pass
    _compareTo = abstractmethod(_compareTo)
    # ...

class Container(Object):
    def __init__(self):
        super(Container,self).__init__()
        self._count=0
    
    @abstractmethod
    def purge(self):pass
    
    @abstractmethod
    def __iter__(self):pass


class Stack(Container):

    def __init__(self):
        super(Stack, self).__init__()

    def getTop(self):
        pass
    getTop = abstractmethod(getTop)

    top = property(
        fget = lambda self: self.getTop())

    def push(self, obj):
        pass
    push = abstractmethod(push)

    def pop(self):
        pass
    pop = abstractmethod(pop)

class Queue(Container):

    def __init__(self):
        super(Queue, self).__init__()

    def getHead(self):
        pass
    getHead = abstractmethod(getHead)

    head = property(
        fget = lambda self: self.getHead())

    def enqueue(self, obj):
        pass
    enqueue = abstractmethod(enqueue)

    def dequeue(self):
        pass
    dequeue = abstractmethod(dequeue)

class Deque(Queue):

    def __init__(self):
        super(Deque, self).__init__()

    def getHead(self):
        pass
    getHead = abstractmethod(getHead)

    head = property(
        fget = lambda self: self.getHead())

    def getTail(self):
        pass
    getTail = abstractmethod(getTail)

    tail = property(
        fget = lambda self: self.getTail())

    def enqueueHead(self, obj):
        pass
    enqueueHead = abstractmethod(enqueueHead)

    def dequeueHead(self):
        return self.dequeue()

    def enqueueTail(self, object):
        self.enqueue(object)

    def dequeueTail(self):
        pass
    dequeueTail = abstractmethod(dequeueTail)




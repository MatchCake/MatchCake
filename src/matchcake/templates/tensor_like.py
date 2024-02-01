import abc


class TensorLike(metaclass=abc.ABCMeta):
    NEEDED_ATTRS = ["shape"]
    NEEDED_MTHDS = ["__array__", "__getitem__"]

    @classmethod
    def __subclasshook__(cls, __subclass):
        has_attrs = all(hasattr(__subclass, attr) for attr in cls.NEEDED_ATTRS)
        has_mthds = all(hasattr(__subclass, mthd) and callable(getattr(__subclass, mthd)) for mthd in cls.NEEDED_MTHDS)
        return has_attrs and has_mthds





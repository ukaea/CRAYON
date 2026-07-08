"""
Useful data structures.
"""

# Standard imports
import enum
import logging

# Local imports

logger = logging.getLogger(__name__)


class CrayonEnum(enum.Enum):
    """
    Enum class with integer values but with __str__ and a case insensitive
    lookup.

    Methods
    -------
    from_string
        Return enum member from name case-insensitively.
    parse
        Parse from object.
    description
        Enum description.
    """

    def __str__(self) -> str:
        """
        Convert to string.

        Returns
        -------
        string : str
            Object value as string.
        """
        return f"{self.name}: {self.value}"

    @classmethod
    def from_string(cls, name: str) -> "CrayonEnum":
        """
        Return enum member from name case-insensitively.

        Parameters
        ----------
        name : str
            Enum name.

        Returns
        -------
        enum : CrayonEnum
            Parsed enum object.

        Raises
        ------
        ValueError
            Name not member of enumeration.
        """
        _name = name.upper()

        if _name in cls.__members__:
            return cls.__members__[_name]
        raise ValueError(
            f"{cls.__name__} has no member '{_name}'. "
            f"Available members: " + ", ".join(x.name for x in cls)
        )

    @classmethod
    def parse(cls, obj, /, *, allow_none: bool = False) -> "CrayonEnum":
        """
        Parse from object.

        Parameters
        ----------
        obj : any
            Object to parse.
        allow_none : bool, optional
            If True, if obj is None return None. Default = False.

        Returns
        -------
        enum : CrayonEnum
            Parsed enum object.
        """
        if allow_none and not obj:
            return None
        if isinstance(obj, CrayonEnum):
            return obj
        return cls.from_string(str(obj))

    @classmethod
    def description(cls) -> str:
        """
        Enum description.

        Returns
        -------
        description : str
            Description.
        """
        return ", ".join(f"{member.name} = {member.value}" for member in cls)


class Dimension:
    """
    Object representing a coordinate dimension.

    Attributes
    ----------
    name : str
        Name of dimension.
    size : int
        Number of components.
    """

    __slots__ = ("name", "size")

    def __init__(self, name: str, size: int):
        """
        Inits Dimension.

        Parameters
        ----------
        name : str
            Name of dimension.
        size : int
            Number of components.
        """
        self.name = name
        self.size = size

    def __hash__(self) -> int:
        """
        Hash object.

        Returns
        -------
        hash : int
            Hash of object.
        """
        return hash(self.name) ^ hash(self.size)

    def __eq__(self, other):
        """
        Compare equality with another object.

        Parameters
        ----------
        other : any
            Another object.

        Returns
        -------
        is_equal : bool
            If objects are equal.
        """
        if isinstance(other, Dimension):
            return (self.name == other.name) and (self.size == other.size)
        return False


class Result:
    """
    Result object like from Rust. Either has value or message explaining error.

    Attributes
    ----------
    message : str
        Any error message.
    value : any
        Return value.
    """

    __slots__ = ("message", "value")

    def __init__(self, value, message: str):
        """
        Inits Result.

        Parameters
        ----------
        value : any
            Value.
        message : str
            Error message.
        """
        self.value = value
        self.message = message

    @classmethod
    def success(cls, value) -> "Result":
        """
        A successful result.

        Parameters
        ----------
        value : any
            Return value.

        Returns
        -------
        result : Result
            Result.
        """
        return cls(value, "")

    @classmethod
    def failure(cls, message: str) -> "Result":
        """
        A failure.

        Parameters
        ----------
        message : str
            Error message.

        Returns
        -------
        result : Result
            Result.
        """
        return cls(0.0, message)

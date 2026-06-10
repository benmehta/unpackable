# unpackable-objects

Zero-dependency unpacking and projection for ordinary Python objects.

This is not a validator, a schema system, or a model framework. It is a tiny decorator and helper API that makes normal Python classes easier to unpack, expand, and project.

The core promise:

> Not a validator. Not a model framework. Just fast, zero-dependency unpacking for real Python objects.

```python
from unpackable import unpackable


@unpackable
class Tick:
    __slots__ = ("symbol", "price", "size")

    def __init__(self, symbol: str, price: float, size: int) -> None:
        self.symbol = symbol
        self.price = price
        self.size = size


tick = Tick("AAPL", 192.4, 100)

symbol, price, size = tick
payload = {**tick}
row = tick.to_tuple()
doc = tick.to_dict()
```

## Why

Many Python libraries focus on validation, parsing, coercion, and schema management. Those are the right tools for APIs and user input. They are also more machinery than you need when you already have trusted objects and just want them to behave nicely at the boundary of your code.

`unpackable-objects` is for codebases where you want:

- The memory behavior of slotted classes.
- The IDE ergonomics of normal Python objects.
- Tuple-style sequence unpacking.
- Dictionary-style `**object` unpacking.
- Recursive conversion for nested objects, lists, tuples, and dictionaries.
- No runtime validation layer and no third-party dependencies.

Use Pydantic, attrs, dataclasses, or msgspec when you want their modeling, validation, or serialization ecosystems. Use this when you want to keep plain classes and add unpacking/projection behavior with one decorator.

## Examples

Market-data rows:

```python
from unpackable import unpackable


@unpackable
class Quote:
    __slots__ = ("symbol", "bid", "ask", "ts_ns")

    def __init__(self, symbol, bid, ask, ts_ns):
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.ts_ns = ts_ns


quote = Quote("AAPL", 192.31, 192.34, 1718047502000000000)

symbol, bid, ask, ts_ns = quote
payload = {**quote}
```

ETL rows with private fields excluded:

```python
@unpackable(exclude={"_raw"}, aliases={"customer_id": "id"})
class CustomerRow:
    __slots__ = ("customer_id", "country", "score", "_raw")

    def __init__(self, customer_id, country, score, raw):
        self.customer_id = customer_id
        self.country = country
        self.score = score
        self._raw = raw


row = CustomerRow("c_123", "US", 0.98, {"debug": True})

assert row.to_dict() == {"id": "c_123", "country": "US", "score": 0.98}
```

Nested objects:

```python
@unpackable
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


@unpackable
class Shape:
    __slots__ = ("name", "points")

    def __init__(self):
        self.name = "triangle"
        self.points = [Point(0, 0), Point(1, 0), Point(0, 1)]


assert Shape().to_dict() == {
    "name": "triangle",
    "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}],
}
```

## Decorator options

```python
@unpackable(
    fields=None,
    exclude={"_cache"},
    include_private=False,
    recursive=True,
    sequence=True,
    mapping=True,
    aliases={"symbol": "s"},
    overwrite=False,
)
class Quote:
    __slots__ = ("symbol", "bid", "ask", "_cache")
```

`fields` is optional. If omitted, the decorator uses annotations and `__slots__` when available. For fully dynamic classes, it falls back to the instance `__dict__`.

## Helper API

You do not have to decorate a class just to project it once. The helper API works on decorated classes, ordinary slotted classes, annotated classes, and dynamic classes.

```python
from unpackable import asdict, astuple, fields


class Point:
    __slots__ = ("x", "y")

    def __init__(self):
        self.x = 1
        self.y = 2


point = Point()

assert fields(Point) == ("x", "y")
assert asdict(point) == {"x": 1, "y": 2}
assert astuple(point) == (1, 2)
```

The decorator is the fastest and most ergonomic path when you control the class. The helpers are useful for one-off projection, migration, debugging, and codebases where method injection is not desired.

## Field Order

Field order is stable and intentional:

- Explicit `fields=(...)` order wins.
- Otherwise annotation order wins.
- Otherwise `__slots__` order wins, including inherited slots from base classes first.
- Dynamic classes use insertion order from the instance `__dict__`.

The same order is used for sequence unpacking, `keys()`, `items()`, `values()`, `to_tuple()`, `to_dict()`, and helper output.

## Performance shape

The decorator computes the field plan once at class decoration time. Slotted and annotated classes get generated methods with their field names baked into the method bodies for `__iter__`, `keys`, `items`, `values`, `__getitem__`, `to_tuple`, and `to_dict`.

Flat primitive fields use a fast path that avoids recursive object inspection. For maximum shallow projection speed, opt into non-recursive mode:

```python
@unpackable(recursive=False)
class Tick:
    __slots__ = ("symbol", "price", "size")
```

In this mode, generated `to_dict()` returns a direct shallow dict literal for complete slotted objects. You can also use the same fast path at runtime with `obj.to_dict(recursive=False)`. Nested objects, dictionaries, lists, tuples, and sets still use recursive conversion when `recursive=True`.

Dynamic non-slotted classes work, but they are not the main performance story. They still need to inspect the instance `__dict__` because their attributes do not exist at class decoration time. If you only need a shallow dictionary from a dynamic object, `obj.__dict__.copy()` is much faster. This package adds stable unpacking protocols, filtering, aliases, and recursive projection.

Run the starter benchmark with:

```bash
python benchmarks/bench_unpackable.py
BENCH_N=500000 python benchmarks/bench_unpackable.py
```

The benchmark always compares decorated objects, manual methods, and `dataclasses.asdict`. If `attrs`, Pydantic, or `msgspec` are installed in your environment, it includes those comparisons automatically. They are benchmark-only comparisons, not runtime dependencies.

Example run on Python 3.12.13:

```text
unpackable slotted tuple(obj)          532.8 ns/call
unpackable slotted {**obj}             951.9 ns/call
unpackable recursive to_dict()         510.3 ns/call
unpackable runtime flat to_dict()      207.9 ns/call
unpackable flat to_dict()              205.1 ns/call
manual slotted to_dict()               152.6 ns/call
dataclasses.asdict(slotted)           2066.7 ns/call
unpackable dynamic to_dict()          2088.0 ns/call
dynamic __dict__.copy()                184.9 ns/call
manual dynamic to_dict()               158.1 ns/call
attrs.asdict(obj)                     1043.7 ns/call
pydantic.model_dump()                 1263.7 ns/call
msgspec.to_builtins(obj)               256.8 ns/call
```

For flat slotted objects, shallow `to_dict()` can be faster than framework dump methods, including `msgspec.to_builtins()` in this benchmark. Hand-written methods, direct `__dict__` copies, and compiled serializers can still win. msgspec remains the right choice for compiled serialization, JSON bytes, and broad schema-driven performance.

To create a benchmark environment with Python 3.12 and all optional comparison libraries:

```bash
conda env create -f environment.yml
conda activate unpackable-bench
python -m pytest -q
python benchmarks/bench_unpackable.py
```

## Not a validator

This package intentionally does not validate or coerce input. That is the point. It is designed for trusted data at scale, not hostile or messy data at the edge of your system.

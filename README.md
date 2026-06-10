# unpackable-objects

Zero-dependency unpacking and projection for ordinary Python objects.

This is not a validator, a schema system, or a model framework. It is a tiny decorator and projection engine that makes normal Python classes easier to unpack, expand, batch, and project.

The core promise:

> Not a validator. Not a model framework. Fast projection for real Python objects.

```python
from unpackable import compile_projector, unpackable


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

projector = compile_projector(Tick, recursive=False)
records = projector.records([tick])
columns = projector.columns([tick])
```

## Why

Many Python libraries focus on validation, parsing, coercion, and schema management. Those are the right tools for APIs and user input. They are also more machinery than you need when you already have trusted objects and just want them to behave nicely at the boundary of your code.

`unpackable-objects` is for codebases where you want:

- The memory behavior of slotted classes.
- The IDE ergonomics of normal Python objects.
- Tuple-style sequence unpacking.
- Dictionary-style `**object` unpacking.
- Recursive conversion for nested objects, lists, tuples, and dictionaries.
- Batch projection into records, tuples, or columns.
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

## Projection Engine

For repeated workloads, compile a projector once and reuse it:

```python
from unpackable import compile_projector


projector = compile_projector(Quote, recursive=False)

records = projector.records(quotes)
tuples = projector.tuples(quotes)
columns = projector.columns(quotes)
```

Records are list-of-dict output:

```python
[
    {"symbol": "AAPL", "bid": 192.31, "ask": 192.34},
    {"symbol": "MSFT", "bid": 410.10, "ask": 410.14},
]
```

Columns are dict-of-list output:

```python
{
    "symbol": ["AAPL", "MSFT"],
    "bid": [192.31, 410.10],
    "ask": [192.34, 410.14],
}
```

Top-level helpers are available when you do not want to hold onto a projector:

```python
from unpackable import to_columns, to_records, to_tuples

records = to_records(quotes, recursive=False)
columns = to_columns(quotes, recursive=False)
tuples = to_tuples(quotes, recursive=False)
```

Optional bridges are dependency-free until called:

```python
from unpackable import to_arrow, to_pandas

df = to_pandas(quotes, recursive=False)      # requires pandas
table = to_arrow(quotes, recursive=False)    # requires pyarrow
```

For column output, uninitialized slotted fields are represented as `None` so every column remains the same length. Single-record `to_dict()` still skips uninitialized slots.

## ML and Data Science Workflows

ML and data science code often starts with ordinary Python objects and eventually needs records, columns, or DataFrames for evaluation, monitoring, plotting, or export.

```python
from unpackable import compile_projector, to_pandas, unpackable


@unpackable(recursive=False)
class Prediction:
    __slots__ = ("user_id", "model_version", "score", "label", "latency_ms")

    def __init__(self, user_id, model_version, score, label, latency_ms):
        self.user_id = user_id
        self.model_version = model_version
        self.score = score
        self.label = label
        self.latency_ms = latency_ms


predictions = [
    Prediction("u1", "model-v4", 0.91, 1, 12.4),
    Prediction("u2", "model-v4", 0.08, 0, 9.7),
    Prediction("u3", "model-v4", 0.63, 1, 14.1),
]

projector = compile_projector(Prediction)

columns = projector.columns(predictions)
records = projector.records(predictions)
```

`columns` is immediately useful for metrics, drift checks, plotting, or DataFrame construction:

```python
{
    "user_id": ["u1", "u2", "u3"],
    "model_version": ["model-v4", "model-v4", "model-v4"],
    "score": [0.91, 0.08, 0.63],
    "label": [1, 0, 1],
    "latency_ms": [12.4, 9.7, 14.1],
}
```

If pandas is installed, you can go straight to a DataFrame:

```python
df = to_pandas(predictions, recursive=False)
```

This is useful when you want readable domain objects inside your inference, simulation, or evaluation code, but tabular output at the analysis boundary.

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
unpackable slotted tuple(obj)          539.3 ns/call
unpackable slotted {**obj}             952.6 ns/call
unpackable recursive to_dict()         486.1 ns/call
unpackable runtime flat to_dict()      212.4 ns/call
unpackable flat to_dict()              210.1 ns/call
projector flat to_dict()               259.3 ns/call
manual slotted to_dict()               154.8 ns/call
dataclasses.asdict(slotted)           2042.5 ns/call
projector records(100)               21189.8 ns/call
projector columns(100)               11708.8 ns/call
helper to_records(100)               23190.3 ns/call
helper to_columns(100)               15724.3 ns/call
attrs.asdict(obj)                     1016.4 ns/call
pydantic.model_dump()                 1244.0 ns/call
msgspec.to_builtins(obj)               260.4 ns/call
msgspec.to_builtins(100)             15086.3 ns/call
```

For flat slotted objects, shallow `to_dict()` can be faster than framework dump methods, including `msgspec.to_builtins()` in this benchmark. For batches, record output and column output are different shapes: msgspec is faster than `projector.records(100)`, while `projector.columns(100)` is faster than msgspec's 100-record builtins conversion because it extracts columnar lists directly. Hand-written methods, direct `__dict__` copies, and compiled serializers can still win. msgspec remains the right choice for compiled serialization, JSON bytes, and broad schema-driven performance.

To create a benchmark environment with Python 3.12 and all optional comparison libraries:

```bash
conda env create -f environment.yml
conda activate unpackable-bench
python -m pytest -q
python benchmarks/bench_unpackable.py
```

## Not a validator

This package intentionally does not validate or coerce input. That is the point. It is designed for trusted data at scale, not hostile or messy data at the edge of your system.

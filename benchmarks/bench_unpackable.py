import dataclasses
import gc
import os
import sys
import timeit
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from unpackable import asdict, astuple, unpackable


@unpackable
class SlottedTick:
    __slots__ = ("symbol", "price", "size")

    def __init__(self, symbol, price, size):
        self.symbol = symbol
        self.price = price
        self.size = size


@unpackable(recursive=False)
class FlatSlottedTick:
    __slots__ = ("symbol", "price", "size")

    def __init__(self, symbol, price, size):
        self.symbol = symbol
        self.price = price
        self.size = size


@unpackable
class DynamicTick:
    def __init__(self, symbol, price, size):
        self.symbol = symbol
        self.price = price
        self.size = size


class ManualSlottedTick:
    __slots__ = ("symbol", "price", "size")

    def __init__(self, symbol, price, size):
        self.symbol = symbol
        self.price = price
        self.size = size

    def __iter__(self):
        yield self.symbol
        yield self.price
        yield self.size

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "price": self.price,
            "size": self.size,
        }


class ManualDynamicTick:
    def __init__(self, symbol, price, size):
        self.symbol = symbol
        self.price = price
        self.size = size

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "price": self.price,
            "size": self.size,
        }


try:
    @dataclasses.dataclass(slots=True)
    class DataclassTick:
        symbol: str
        price: float
        size: int

    DATACLASS_LABEL = "dataclasses.asdict(slotted)"
except TypeError:
    @dataclasses.dataclass
    class DataclassTick:
        symbol: str
        price: float
        size: int

    DATACLASS_LABEL = "dataclasses.asdict(dynamic)"


def _optional_cases() -> Dict[str, Tuple[Any, List[Tuple[str, str]]]]:
    cases: Dict[str, Tuple[Any, List[Tuple[str, str]]]] = {}

    try:
        import attr

        @attr.define(slots=True)
        class AttrsTick:
            symbol: str
            price: float
            size: int

        cases["attrs"] = (
            AttrsTick("AAPL", 192.4, 100),
            [("attrs.asdict(obj)", "attr.asdict(obj)")],
        )
        globals()["attr"] = attr
    except ImportError:
        pass

    try:
        from pydantic import BaseModel

        class PydanticTick(BaseModel):
            symbol: str
            price: float
            size: int

        model = PydanticTick(symbol="AAPL", price=192.4, size=100)
        dump_name = "model_dump" if hasattr(model, "model_dump") else "dict"
        cases["pydantic"] = (
            model,
            [(f"pydantic.{dump_name}()", f"obj.{dump_name}()")],
        )
    except ImportError:
        pass

    try:
        import msgspec

        class MsgspecTick(msgspec.Struct):
            symbol: str
            price: float
            size: int

        cases["msgspec"] = (
            MsgspecTick("AAPL", 192.4, 100),
            [("msgspec.to_builtins(obj)", "msgspec.to_builtins(obj)")],
        )
        globals()["msgspec"] = msgspec
    except ImportError:
        pass

    return cases


def bench(name: str, statement: str, setup: str, number: int) -> float:
    repeat = 5
    timings = timeit.repeat(
        statement,
        setup=setup,
        number=number,
        repeat=repeat,
        globals=globals(),
    )
    best = min(timings)
    return best / number * 1_000_000_000


def run_case(name: str, statement: str, setup: str, number: int) -> None:
    per_call_ns = bench(name, statement, setup, number)
    print(f"{name:34s} {per_call_ns:9.1f} ns/call")


if __name__ == "__main__":
    gc.disable()
    number = int(os.environ.get("BENCH_N", "100000"))

    setup = (
        "from __main__ import "
        "SlottedTick, FlatSlottedTick, DynamicTick, ManualSlottedTick, ManualDynamicTick, DataclassTick, "
        "DATACLASS_LABEL, asdict, astuple, dataclasses; "
        "s = SlottedTick('AAPL', 192.4, 100); "
        "f = FlatSlottedTick('AAPL', 192.4, 100); "
        "d = DynamicTick('AAPL', 192.4, 100); "
        "m = ManualSlottedTick('AAPL', 192.4, 100); "
        "md = ManualDynamicTick('AAPL', 192.4, 100); "
        "dc = DataclassTick('AAPL', 192.4, 100)"
    )

    print(f"Iterations: {number:,}")
    run_case("unpackable slotted tuple(obj)", "tuple(s)", setup, number)
    run_case("unpackable slotted {**obj}", "{**s}", setup, number)
    run_case("unpackable recursive to_dict()", "s.to_dict()", setup, number)
    run_case("unpackable runtime flat to_dict()", "s.to_dict(recursive=False)", setup, number)
    run_case("unpackable flat to_dict()", "f.to_dict()", setup, number)
    run_case("unpackable flat to_tuple()", "f.to_tuple()", setup, number)
    run_case("helper asdict(slotted)", "asdict(s)", setup, number)
    run_case("helper astuple(slotted)", "astuple(s)", setup, number)
    run_case("manual slotted tuple(obj)", "tuple(m)", setup, number)
    run_case("manual slotted to_dict()", "m.to_dict()", setup, number)
    run_case(DATACLASS_LABEL, "dataclasses.asdict(dc)", setup, number)
    run_case("unpackable dynamic tuple(obj)", "tuple(d)", setup, number)
    run_case("unpackable dynamic {**obj}", "{**d}", setup, number)
    run_case("unpackable dynamic to_dict()", "d.to_dict()", setup, number)
    run_case("dynamic __dict__.copy()", "d.__dict__.copy()", setup, number)
    run_case("manual dynamic to_dict()", "md.to_dict()", setup, number)

    optional = _optional_cases()
    for package_name, (obj, package_cases) in optional.items():
        globals()["obj"] = obj
        for label, statement in package_cases:
            run_case(label, statement, "", number)

    missing = {"attrs", "pydantic", "msgspec"} - set(optional)
    if missing:
        print()
        print("Skipped optional comparisons:", ", ".join(sorted(missing)))

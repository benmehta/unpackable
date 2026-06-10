import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unpackable import (
    asdict,
    astuple,
    compile_projector,
    fields,
    is_unpackable,
    to_columns,
    to_jsonable,
    to_records,
    to_tuples,
    unpackable,
)


class UnpackableTests(unittest.TestCase):
    def test_slotted_class_supports_sequence_and_mapping_unpacking(self):
        @unpackable
        class Tick:
            __slots__ = ("symbol", "price", "size")

            def __init__(self):
                self.symbol = "AAPL"
                self.price = 192.4
                self.size = 100

        tick = Tick()

        self.assertEqual(tuple(tick), ("AAPL", 192.4, 100))
        self.assertEqual({**tick}, {"symbol": "AAPL", "price": 192.4, "size": 100})
        self.assertEqual(tick["price"], 192.4)
        self.assertEqual(tick.keys(), ("symbol", "price", "size"))
        self.assertEqual(fields(Tick), ("symbol", "price", "size"))
        self.assertTrue(is_unpackable(tick))

    def test_dynamic_class_falls_back_to_instance_dict(self):
        @unpackable
        class Row:
            def __init__(self):
                self.a = 1
                self.b = 2

        row = Row()

        self.assertEqual(tuple(row), (1, 2))
        self.assertEqual({**row}, {"a": 1, "b": 2})
        self.assertEqual(fields(row), ("a", "b"))

    def test_field_order_prefers_annotations_then_slots(self):
        @unpackable
        class Annotated:
            third: int
            first: int
            second: int

            def __init__(self):
                self.first = 1
                self.second = 2
                self.third = 3

        @unpackable
        class Slotted:
            __slots__ = ("third", "first", "second")

            def __init__(self):
                self.first = 1
                self.second = 2
                self.third = 3

        self.assertEqual(fields(Annotated), ("third", "first", "second"))
        self.assertEqual(tuple(Annotated()), (3, 1, 2))
        self.assertEqual(fields(Slotted), ("third", "first", "second"))
        self.assertEqual(tuple(Slotted()), (3, 1, 2))

    def test_exclude_and_private_filtering(self):
        @unpackable(exclude={"skip"})
        class Payload:
            __slots__ = ("keep", "skip", "_private")

            def __init__(self):
                self.keep = 1
                self.skip = 2
                self._private = 3

        payload = Payload()

        self.assertEqual(payload.to_dict(), {"keep": 1})
        self.assertNotIn("skip", payload)
        self.assertNotIn("_private", payload)

    def test_include_private(self):
        @unpackable(include_private=True)
        class Payload:
            __slots__ = ("_value",)

            def __init__(self):
                self._value = 1

        self.assertEqual(Payload().to_dict(), {"_value": 1})

    def test_recursive_conversion(self):
        @unpackable
        class Point:
            __slots__ = ("x", "y")

            def __init__(self, x, y):
                self.x = x
                self.y = y

        @unpackable
        class Shape:
            __slots__ = ("name", "points", "meta")

            def __init__(self):
                self.name = "tri"
                self.points = [Point(1, 2), Point(3, 4)]
                self.meta = {"origin": Point(0, 0)}

        self.assertEqual(
            asdict(Shape()),
            {
                "name": "tri",
                "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
                "meta": {"origin": {"x": 0, "y": 0}},
            },
        )

    def test_recursive_conversion_can_be_disabled(self):
        @unpackable
        class Point:
            __slots__ = ("x", "y")

            def __init__(self):
                self.x = 1
                self.y = 2

        @unpackable
        class Box:
            __slots__ = ("point",)

            def __init__(self):
                self.point = Point()

        box = Box()

        self.assertIs(box.to_dict(recursive=False)["point"], box.point)
        self.assertEqual(box.to_dict(recursive=True), {"point": {"x": 1, "y": 2}})

    def test_recursive_false_decorator_is_shallow_by_default(self):
        @unpackable
        class Point:
            __slots__ = ("x", "y")

            def __init__(self):
                self.x = 1
                self.y = 2

        @unpackable(recursive=False)
        class Box:
            __slots__ = ("point", "points", "meta")

            def __init__(self):
                self.point = Point()
                self.points = [Point()]
                self.meta = {"point": Point()}

        box = Box()
        payload = box.to_dict()

        self.assertIs(payload["point"], box.point)
        self.assertIs(payload["points"], box.points)
        self.assertIs(payload["meta"], box.meta)
        self.assertEqual(
            box.to_dict(recursive=True),
            {
                "point": {"x": 1, "y": 2},
                "points": [{"x": 1, "y": 2}],
                "meta": {"point": {"x": 1, "y": 2}},
            },
        )

    def test_recursive_false_skips_uninitialized_slots(self):
        @unpackable(recursive=False)
        class Partial:
            __slots__ = ("ready", "missing")

            def __init__(self):
                self.ready = True

        partial = Partial()

        self.assertEqual(partial.to_dict(), {"ready": True})
        self.assertEqual(partial.to_tuple(), (True,))

    def test_aliases_for_static_fields(self):
        @unpackable(aliases={"symbol": "s"})
        class Tick:
            __slots__ = ("symbol",)

            def __init__(self):
                self.symbol = "AAPL"

        tick = Tick()

        self.assertEqual(tick.keys(), ("s",))
        self.assertEqual(tick["s"], "AAPL")
        self.assertEqual({**tick}, {"s": "AAPL"})

    def test_aliases_for_dynamic_fields(self):
        @unpackable(aliases={"symbol": "s"})
        class Tick:
            def __init__(self):
                self.symbol = "AAPL"

        tick = Tick()

        self.assertEqual(tick.keys(), ("s",))
        self.assertEqual(tick["s"], "AAPL")
        self.assertEqual({**tick}, {"s": "AAPL"})

    def test_uninitialized_slot_is_skipped(self):
        @unpackable
        class Partial:
            __slots__ = ("ready", "missing")

            def __init__(self):
                self.ready = True

        partial = Partial()

        self.assertEqual(partial.to_dict(), {"ready": True})
        with self.assertRaises(KeyError):
            partial["missing"]

    def test_helpers(self):
        @unpackable
        class Pair:
            __slots__ = ("left", "right")

            def __init__(self):
                self.left = (1, 2)
                self.right = {3, 4}

        pair = Pair()

        self.assertEqual(astuple(pair), ((1, 2), {3, 4}))
        self.assertEqual(to_jsonable(pair), {"left": [1, 2], "right": [3, 4]})

    def test_helpers_project_undecorated_slotted_objects(self):
        class Point:
            __slots__ = ("x", "y", "_cache")

            def __init__(self):
                self.x = 1
                self.y = 2
                self._cache = 3

        point = Point()

        self.assertEqual(fields(Point), ("x", "y"))
        self.assertEqual(asdict(point), {"x": 1, "y": 2})
        self.assertEqual(astuple(point), (1, 2))
        self.assertEqual(asdict(point, include_private=True), {"x": 1, "y": 2, "_cache": 3})

    def test_helpers_project_undecorated_dynamic_objects(self):
        class Row:
            def __init__(self):
                self.a = 1
                self.b = 2

        row = Row()

        self.assertEqual(fields(row), ("a", "b"))
        self.assertEqual(asdict(row, aliases={"a": "alpha"}), {"alpha": 1, "b": 2})
        self.assertEqual(astuple(row, fields=("b", "a")), (2, 1))

    def test_recursive_helpers_project_nested_undecorated_objects(self):
        class Point:
            __slots__ = ("x", "y")

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class Shape:
            __slots__ = ("points",)

            def __init__(self):
                self.points = [Point(1, 2), Point(3, 4)]

        self.assertEqual(
            asdict(Shape()),
            {"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
        )

    def test_compiled_projector_projects_records_tuples_and_columns(self):
        @unpackable(recursive=False)
        class Tick:
            __slots__ = ("symbol", "price", "size")

            def __init__(self, symbol, price, size):
                self.symbol = symbol
                self.price = price
                self.size = size

        ticks = [Tick("AAPL", 192.4, 100), Tick("MSFT", 410.2, 50)]
        projector = compile_projector(Tick)

        self.assertEqual(projector.fields, ("symbol", "price", "size"))
        self.assertEqual(
            projector.records(ticks),
            [
                {"symbol": "AAPL", "price": 192.4, "size": 100},
                {"symbol": "MSFT", "price": 410.2, "size": 50},
            ],
        )
        self.assertEqual(
            projector.tuples(ticks),
            [("AAPL", 192.4, 100), ("MSFT", 410.2, 50)],
        )
        self.assertEqual(
            projector.columns(ticks),
            {
                "symbol": ["AAPL", "MSFT"],
                "price": [192.4, 410.2],
                "size": [100, 50],
            },
        )

    def test_top_level_batch_projection_helpers(self):
        class Tick:
            __slots__ = ("symbol", "price", "size")

            def __init__(self, symbol, price, size):
                self.symbol = symbol
                self.price = price
                self.size = size

        ticks = [Tick("AAPL", 192.4, 100), Tick("MSFT", 410.2, 50)]

        self.assertEqual(
            to_records(ticks, aliases={"symbol": "s"}, recursive=False),
            [
                {"s": "AAPL", "price": 192.4, "size": 100},
                {"s": "MSFT", "price": 410.2, "size": 50},
            ],
        )
        self.assertEqual(
            to_tuples(ticks, fields=("price", "symbol"), recursive=False),
            [(192.4, "AAPL"), (410.2, "MSFT")],
        )
        self.assertEqual(
            to_columns(ticks, fields=("symbol", "size"), recursive=False),
            {"symbol": ["AAPL", "MSFT"], "size": [100, 50]},
        )

    def test_projector_columns_use_none_for_missing_slots(self):
        class Partial:
            __slots__ = ("ready", "missing")

            def __init__(self, ready, set_missing=False):
                self.ready = ready
                if set_missing:
                    self.missing = "set"

        rows = [Partial(True), Partial(False, set_missing=True)]
        projector = compile_projector(Partial, recursive=False)

        self.assertEqual(
            projector.columns(rows),
            {"ready": [True, False], "missing": [None, "set"]},
        )
        self.assertEqual(projector.records(rows), [{"ready": True}, {"ready": False, "missing": "set"}])

    def test_dynamic_projector_columns_pad_varying_fields(self):
        class Row:
            def __init__(self, **values):
                vars(self).update(values)

        rows = [Row(a=1), Row(a=2, b=3), Row(b=4)]

        self.assertEqual(
            to_columns(rows, recursive=False),
            {"a": [1, 2, None], "b": [None, 3, 4]},
        )

    def test_projector_recursive_mode(self):
        @unpackable
        class Point:
            __slots__ = ("x", "y")

            def __init__(self):
                self.x = 1
                self.y = 2

        @unpackable(recursive=False)
        class Box:
            __slots__ = ("point",)

            def __init__(self):
                self.point = Point()

        box = Box()
        projector = compile_projector(Box)

        self.assertIs(projector.to_dict(box)["point"], box.point)
        self.assertEqual(
            compile_projector(Box, recursive=True).to_dict(box),
            {"point": {"x": 1, "y": 2}},
        )


if __name__ == "__main__":
    unittest.main()

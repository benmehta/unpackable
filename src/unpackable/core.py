from __future__ import annotations

from dataclasses import dataclass
import keyword
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")

_SKIP_SLOTS = frozenset({"__dict__", "__weakref__"})
_ATOMIC_TYPES = frozenset({type(None), bool, int, float, str, bytes})


class SupportsUnpackable(Protocol):
    def __iter__(self) -> Iterator[Any]:
        ...

    def keys(self) -> Sequence[str]:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def to_dict(self, *, recursive: Optional[bool] = None) -> Dict[str, Any]:
        ...

    def to_tuple(self, *, recursive: Optional[bool] = None) -> Tuple[Any, ...]:
        ...


@dataclass(frozen=True)
class _Plan:
    source_names: Tuple[str, ...]
    public_names: Tuple[str, ...]
    key_to_source: Mapping[str, str]
    aliases: Mapping[str, str]
    source_by_alias: Mapping[str, str]
    exclude: FrozenSet[str]
    include_private: bool
    recursive: bool
    static: bool


def _unique(names: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _slot_names(cls: Type[Any]) -> Tuple[str, ...]:
    names: List[str] = []
    for base in reversed(cls.__mro__):
        raw_slots = getattr(base, "__slots__", ())
        if isinstance(raw_slots, str):
            slots = (raw_slots,)
        elif isinstance(raw_slots, Mapping):
            slots = tuple(raw_slots)
        else:
            slots = tuple(raw_slots)

        for slot in slots:
            if slot not in _SKIP_SLOTS:
                names.append(slot)
    return _unique(names)


def _annotation_names(cls: Type[Any]) -> Tuple[str, ...]:
    names: List[str] = []
    for base in reversed(cls.__mro__):
        annotations = getattr(base, "__annotations__", {})
        names.extend(annotations)
    return _unique(names)


def _public_name(source_name: str, aliases: Mapping[str, str]) -> str:
    return aliases.get(source_name, source_name)


def _keep_name(name: str, exclude: FrozenSet[str], include_private: bool) -> bool:
    if name in exclude:
        return False
    if not include_private and name.startswith("_"):
        return False
    return True


def _build_plan(
    cls: Type[Any],
    *,
    fields: Optional[Iterable[str]],
    exclude: Iterable[str],
    include_private: bool,
    recursive: bool,
    aliases: Optional[Mapping[str, str]],
) -> _Plan:
    exclude_set = frozenset(exclude)
    alias_map = dict(aliases or {})

    explicit_fields = tuple(fields) if fields is not None else ()
    discovered = explicit_fields or _annotation_names(cls) or _slot_names(cls)
    static = bool(discovered)

    source_names = tuple(
        name for name in discovered if _keep_name(name, exclude_set, include_private)
    )
    public_names = tuple(_public_name(name, alias_map) for name in source_names)

    if len(set(public_names)) != len(public_names):
        raise ValueError(f"{cls.__name__} has duplicate unpacked field names")

    return _Plan(
        source_names=source_names,
        public_names=public_names,
        key_to_source=dict(zip(public_names, source_names)),
        aliases=alias_map,
        source_by_alias={public: source for source, public in alias_map.items()},
        exclude=exclude_set,
        include_private=include_private,
        recursive=recursive,
        static=static,
    )


def _convert(value: Any, *, jsonable: bool, recursive: bool) -> Any:
    if not recursive:
        return value

    if type(value) in _ATOMIC_TYPES:
        return value

    if is_unpackable(value):
        if jsonable:
            return value.to_jsonable(recursive=True)
        return value.to_dict(recursive=True)

    if _is_plain_projectable(value):
        return _project_dict(value, recursive=True, jsonable=jsonable)

    if isinstance(value, dict):
        return {
            key: _convert(item, jsonable=jsonable, recursive=True)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_convert(item, jsonable=jsonable, recursive=True) for item in value]

    if isinstance(value, tuple):
        converted = [_convert(item, jsonable=jsonable, recursive=True) for item in value]
        return converted if jsonable else tuple(converted)

    if isinstance(value, set):
        converted = [_convert(item, jsonable=jsonable, recursive=True) for item in value]
        return converted if jsonable else set(converted)

    return value


def _convert_field(value: Any, *, jsonable: bool, recursive: bool) -> Any:
    if not recursive or type(value) in _ATOMIC_TYPES:
        return value
    return _convert(value, jsonable=jsonable, recursive=True)


def _attr_expr(name: str) -> str:
    if name.isidentifier() and not keyword.iskeyword(name):
        return f"self.{name}"
    return f"getattr(self, {name!r})"


def _missing_key(obj: Any, key: str) -> KeyError:
    return KeyError(f"{type(obj).__name__!s} has no unpacked field {key!r}")


def _iter_static_items(obj: Any, plan: _Plan) -> Iterator[Tuple[str, Any]]:
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        try:
            yield public_name, getattr(obj, source_name)
        except AttributeError:
            continue


def _iter_dynamic_items(obj: Any, plan: _Plan) -> Iterator[Tuple[str, Any]]:
    try:
        instance_vars = vars(obj)
    except TypeError:
        return

    for source_name, value in instance_vars.items():
        if _keep_name(source_name, plan.exclude, plan.include_private):
            yield _public_name(source_name, plan.aliases), value


def _plan_items(obj: Any, plan: _Plan) -> Iterator[Tuple[str, Any]]:
    if plan.static:
        return _iter_static_items(obj, plan)
    return _iter_dynamic_items(obj, plan)


def _is_plain_projectable(obj: Any) -> bool:
    if isinstance(obj, type):
        return False
    if is_unpackable(obj):
        return False
    if type(obj).__module__ == "builtins":
        return False
    cls = type(obj)
    return bool(_annotation_names(cls) or _slot_names(cls) or hasattr(obj, "__dict__"))


def _plain_plan(
    obj_or_cls: Any,
    *,
    fields: Optional[Iterable[str]],
    exclude: Iterable[str],
    include_private: bool,
    recursive: bool,
    aliases: Optional[Mapping[str, str]],
) -> _Plan:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    return _build_plan(
        cls,
        fields=fields,
        exclude=exclude,
        include_private=include_private,
        recursive=recursive,
        aliases=aliases,
    )


def _project_dict(
    obj: Any,
    *,
    fields: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = (),
    include_private: bool = False,
    recursive: bool = True,
    aliases: Optional[Mapping[str, str]] = None,
    jsonable: bool = False,
) -> Dict[str, Any]:
    plan = _plain_plan(
        obj,
        fields=fields,
        exclude=exclude,
        include_private=include_private,
        recursive=recursive,
        aliases=aliases,
    )
    return {
        key: _convert_field(value, jsonable=jsonable, recursive=recursive)
        for key, value in _plan_items(obj, plan)
    }


def _project_tuple(
    obj: Any,
    *,
    fields: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = (),
    include_private: bool = False,
    recursive: bool = True,
    aliases: Optional[Mapping[str, str]] = None,
) -> Tuple[Any, ...]:
    plan = _plain_plan(
        obj,
        fields=fields,
        exclude=exclude,
        include_private=include_private,
        recursive=recursive,
        aliases=aliases,
    )
    return tuple(
        _convert_field(value, jsonable=False, recursive=recursive)
        for _, value in _plan_items(obj, plan)
    )


def _make_static_methods(plan: _Plan) -> Dict[str, Callable[..., Any]]:
    lines = [
        "def __iter__(self):",
    ]
    if plan.source_names:
        for source_name in plan.source_names:
            attr = _attr_expr(source_name)
            lines.extend(
                [
                    "    try:",
                    f"        yield {attr}",
                    "    except AttributeError:",
                    "        pass",
                ]
            )
    else:
        lines.append("    return iter(())")

    lines.extend(
        [
            "",
            "def __len__(self):",
            "    try:",
        ]
    )
    for source_name in plan.source_names:
        lines.append(f"        {_attr_expr(source_name)}")
    lines.extend(
            [
                "        return _field_count",
                "    except AttributeError:",
                "        count = 0",
            ]
        )
    for source_name in plan.source_names:
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "    try:",
                f"        {attr}",
                "        count += 1",
                "    except AttributeError:",
                "        pass",
            ]
        )
    lines.append("    return count")

    lines.extend(["", "def keys(self):", "    try:"])
    for source_name in plan.source_names:
        lines.append(f"        {_attr_expr(source_name)}")
    lines.extend(
            [
                "        return _public_names",
                "    except AttributeError:",
                "        out = []",
            ]
        )
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "    try:",
                f"        {attr}",
                f"        out.append({public_name!r})",
                "    except AttributeError:",
                "        pass",
            ]
        )
    lines.append("    return tuple(out)")

    lines.extend(["", "def items(self):", "    try:"])
    for source_name in plan.source_names:
        lines.append(f"        {_attr_expr(source_name)}")
    if plan.public_names:
        pairs = ", ".join(
            f"({public_name!r}, {_attr_expr(source_name)})"
            for public_name, source_name in zip(plan.public_names, plan.source_names)
        )
        lines.append(f"        return ({pairs},)")
    else:
        lines.append("        return ()")
    lines.extend(
            [
                "    except AttributeError:",
                "        out = []",
            ]
        )
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "    try:",
                f"        value = {attr}",
                f"        out.append(({public_name!r}, value))",
                "    except AttributeError:",
                "        pass",
            ]
        )
    lines.append("    return tuple(out)")

    lines.extend(["", "def values(self):", "    try:"])
    for source_name in plan.source_names:
        lines.append(f"        {_attr_expr(source_name)}")
    if plan.source_names:
        values = ", ".join(_attr_expr(name) for name in plan.source_names)
        lines.append(f"        return ({values},)")
    else:
        lines.append("        return ()")
    lines.extend(
            [
                "    except AttributeError:",
                "        out = []",
            ]
        )
    for source_name in plan.source_names:
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "    try:",
                f"        out.append({attr})",
                "    except AttributeError:",
                "        pass",
            ]
        )
    lines.append("    return tuple(out)")

    lines.extend(
        [
            "",
            "def __contains__(self, key):",
            "    if not isinstance(key, str):",
            "        return False",
        ]
    )
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        attr = _attr_expr(source_name)
        lines.extend(
            [
                f"    if key == {public_name!r}:",
                "        try:",
                f"            {attr}",
                "            return True",
                "        except AttributeError:",
                "            return False",
            ]
        )
    lines.append("    return False")

    lines.extend(["", "def __getitem__(self, key):"])
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        attr = _attr_expr(source_name)
        lines.extend(
            [
                f"    if key == {public_name!r}:",
                "        try:",
                f"            return {attr}",
                "        except AttributeError:",
                "            raise _missing_key(self, key)",
            ]
        )
    lines.append("    raise _missing_key(self, key)")

    lines.extend(
        [
            "",
            "def get(self, key, default=None):",
            "    try:",
            "        return self[key]",
            "    except KeyError:",
            "        return default",
            "",
            "def to_dict(self, *, recursive=None):",
            "    should_recurse = _plan_recursive if recursive is None else recursive",
            "    if not should_recurse:",
            "        try:",
        ]
    )
    if plan.public_names:
        shallow_entries = ", ".join(
            f"{public_name!r}: {_attr_expr(source_name)}"
            for public_name, source_name in zip(plan.public_names, plan.source_names)
        )
        lines.append(f"            return {{{shallow_entries}}}")
    else:
        lines.append("            return {}")
    lines.extend(
        [
            "        except AttributeError:",
            "            out = {}",
        ]
    )
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "            try:",
                f"                out[{public_name!r}] = {attr}",
                "            except AttributeError:",
                "                pass",
            ]
        )
    lines.extend(
        [
            "            return out",
            "    try:",
        ]
    )
    for index, source_name in enumerate(plan.source_names):
        lines.append(f"        value_{index} = {_attr_expr(source_name)}")
    if plan.public_names:
        entries = ", ".join(
            f"{public_name!r}: _convert_field(value_{index}, jsonable=False, recursive=should_recurse)"
            for index, public_name in enumerate(plan.public_names)
        )
        lines.append(f"        return {{{entries}}}")
    else:
        lines.append("        return {}")
    lines.extend(
            [
                "    except AttributeError:",
                "        out = {}",
            ]
        )
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "    try:",
                f"        value = {attr}",
                f"        out[{public_name!r}] = _convert_field(value, jsonable=False, recursive=should_recurse)",
                "    except AttributeError:",
                "        pass",
            ]
        )
    lines.append("    return out")

    lines.extend(
        [
            "",
            "def to_tuple(self, *, recursive=None):",
            "    should_recurse = _plan_recursive if recursive is None else recursive",
            "    if not should_recurse:",
            "        try:",
        ]
    )
    if plan.source_names:
        shallow_values = ", ".join(_attr_expr(name) for name in plan.source_names)
        lines.append(f"            return ({shallow_values},)")
    else:
        lines.append("            return ()")
    lines.extend(
        [
            "        except AttributeError:",
            "            out = []",
        ]
    )
    for source_name in plan.source_names:
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "            try:",
                f"                out.append({attr})",
                "            except AttributeError:",
                "                pass",
            ]
        )
    lines.extend(
        [
            "            return tuple(out)",
            "    try:",
        ]
    )
    for index, source_name in enumerate(plan.source_names):
        lines.append(f"        value_{index} = {_attr_expr(source_name)}")
    if plan.source_names:
        values = ", ".join(
            f"_convert_field(value_{index}, jsonable=False, recursive=should_recurse)"
            for index, _ in enumerate(plan.source_names)
        )
        lines.append(f"        return ({values},)")
    else:
        lines.append("        return ()")
    lines.extend(
            [
                "    except AttributeError:",
                "        out = []",
            ]
        )
    for source_name in plan.source_names:
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "    try:",
                f"        value = {attr}",
                "        out.append(_convert_field(value, jsonable=False, recursive=should_recurse))",
                "    except AttributeError:",
                "        pass",
            ]
        )
    lines.append("    return tuple(out)")

    lines.extend(
        [
            "",
            "def to_jsonable(self, *, recursive=None):",
            "    should_recurse = _plan_recursive if recursive is None else recursive",
            "    try:",
        ]
    )
    for index, source_name in enumerate(plan.source_names):
        lines.append(f"        value_{index} = {_attr_expr(source_name)}")
    if plan.public_names:
        entries = ", ".join(
            f"{public_name!r}: _convert_field(value_{index}, jsonable=True, recursive=should_recurse)"
            for index, public_name in enumerate(plan.public_names)
        )
        lines.append(f"        return {{{entries}}}")
    else:
        lines.append("        return {}")
    lines.extend(
            [
                "    except AttributeError:",
                "        out = {}",
            ]
        )
    for public_name, source_name in zip(plan.public_names, plan.source_names):
        attr = _attr_expr(source_name)
        lines.extend(
            [
                "    try:",
                f"        value = {attr}",
                f"        out[{public_name!r}] = _convert_field(value, jsonable=True, recursive=should_recurse)",
                "    except AttributeError:",
                "        pass",
            ]
        )
    lines.append("    return out")

    namespace = {
        "_convert": _convert,
        "_convert_field": _convert_field,
        "_missing_key": _missing_key,
        "_field_count": len(plan.source_names),
        "_plan_recursive": plan.recursive,
        "_public_names": plan.public_names,
        "AttributeError": AttributeError,
        "KeyError": KeyError,
        "getattr": getattr,
        "hasattr": hasattr,
        "isinstance": isinstance,
        "iter": iter,
        "str": str,
        "tuple": tuple,
    }
    local_namespace: Dict[str, Callable[..., Any]] = {}
    exec("\n".join(lines), namespace, local_namespace)

    return {
        "__iter__": local_namespace["__iter__"],
        "__len__": local_namespace["__len__"],
        "__contains__": local_namespace["__contains__"],
        "__getitem__": local_namespace["__getitem__"],
        "keys": local_namespace["keys"],
        "items": local_namespace["items"],
        "values": local_namespace["values"],
        "get": local_namespace["get"],
        "to_dict": local_namespace["to_dict"],
        "to_tuple": local_namespace["to_tuple"],
        "to_jsonable": local_namespace["to_jsonable"],
    }


def _make_dynamic_methods(plan: _Plan) -> Dict[str, Callable[..., Any]]:
    def __iter__(self: Any) -> Iterator[Any]:
        for _, value in _iter_dynamic_items(self, plan):
            yield value

    def __len__(self: Any) -> int:
        try:
            return sum(
                1
                for name in vars(self)
                if _keep_name(name, plan.exclude, plan.include_private)
            )
        except TypeError:
            return 0

    def keys(self: Any) -> Tuple[str, ...]:
        return tuple(key for key, _ in _iter_dynamic_items(self, plan))

    def items(self: Any) -> Tuple[Tuple[str, Any], ...]:
        return tuple(_iter_dynamic_items(self, plan))

    def values(self: Any) -> Tuple[Any, ...]:
        return tuple(value for _, value in _iter_dynamic_items(self, plan))

    def __contains__(self: Any, key: object) -> bool:
        if not isinstance(key, str):
            return False
        source_key = plan.source_by_alias.get(key, key)
        try:
            return source_key in vars(self) and _keep_name(
                source_key, plan.exclude, plan.include_private
            )
        except TypeError:
            return False

    def __getitem__(self: Any, key: str) -> Any:
        source_key = plan.source_by_alias.get(key, key)
        try:
            value = vars(self)[source_key]
        except (KeyError, TypeError):
            raise _missing_key(self, key)

        if not _keep_name(source_key, plan.exclude, plan.include_private):
            raise _missing_key(self, key)
        return value

    def get(self: Any, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self: Any, *, recursive: Optional[bool] = None) -> Dict[str, Any]:
        should_recurse = plan.recursive if recursive is None else recursive
        return {
            key: _convert_field(value, jsonable=False, recursive=should_recurse)
            for key, value in _iter_dynamic_items(self, plan)
        }

    def to_tuple(self: Any, *, recursive: Optional[bool] = None) -> Tuple[Any, ...]:
        should_recurse = plan.recursive if recursive is None else recursive
        return tuple(
            _convert_field(value, jsonable=False, recursive=should_recurse)
            for _, value in _iter_dynamic_items(self, plan)
        )

    def to_jsonable(self: Any, *, recursive: Optional[bool] = None) -> Dict[str, Any]:
        should_recurse = plan.recursive if recursive is None else recursive
        return {
            key: _convert_field(value, jsonable=True, recursive=should_recurse)
            for key, value in _iter_dynamic_items(self, plan)
        }

    methods: Dict[str, Callable[..., Any]] = {
        "__len__": __len__,
        "__contains__": __contains__,
        "to_dict": to_dict,
        "to_tuple": to_tuple,
        "to_jsonable": to_jsonable,
        "items": items,
        "values": values,
        "get": get,
    }

    return {
        **methods,
        "__iter__": __iter__,
        "keys": keys,
        "__getitem__": __getitem__,
    }


def _make_methods(plan: _Plan) -> Dict[str, Callable[..., Any]]:
    if plan.static:
        return _make_static_methods(plan)
    return _make_dynamic_methods(plan)


def _install(cls: Type[T], methods: Mapping[str, Callable[..., Any]], overwrite: bool) -> Type[T]:
    protected = [
        name
        for name in methods
        if name in cls.__dict__ and not (overwrite or name.startswith("__unpackable_"))
    ]
    if protected:
        names = ", ".join(sorted(protected))
        raise TypeError(
            f"{cls.__name__} already defines {names}; pass overwrite=True to replace"
        )

    for name, method in methods.items():
        if overwrite or name not in cls.__dict__:
            setattr(cls, name, method)
    return cls


@overload
def unpackable(cls: Type[T]) -> Type[T]:
    ...


@overload
def unpackable(
    *,
    fields: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = (),
    include_private: bool = False,
    recursive: bool = True,
    sequence: bool = True,
    mapping: bool = True,
    aliases: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
) -> Callable[[Type[T]], Type[T]]:
    ...


def unpackable(
    cls: Optional[Type[T]] = None,
    *,
    fields: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = (),
    include_private: bool = False,
    recursive: bool = True,
    sequence: bool = True,
    mapping: bool = True,
    aliases: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Decorate a class with fast sequence, mapping, and conversion helpers."""

    def decorate(target: Type[T]) -> Type[T]:
        plan = _build_plan(
            target,
            fields=fields,
            exclude=exclude,
            include_private=include_private,
            recursive=recursive,
            aliases=aliases,
        )
        methods = _make_methods(plan)

        if not sequence:
            methods.pop("__iter__", None)
        if not mapping:
            methods.pop("keys", None)
            methods.pop("__getitem__", None)

        setattr(target, "__unpackable_plan__", plan)
        if plan.static:
            setattr(target, "__match_args__", plan.source_names)

        return _install(target, methods, overwrite=overwrite)

    if cls is not None:
        return decorate(cls)
    return decorate


def is_unpackable(obj: Any) -> bool:
    return hasattr(obj, "__unpackable_plan__")


def fields(
    obj_or_cls: Any,
    *,
    exclude: Iterable[str] = (),
    include_private: bool = False,
    aliases: Optional[Mapping[str, str]] = None,
) -> Tuple[str, ...]:
    plan = getattr(obj_or_cls, "__unpackable_plan__", None)
    if plan is None:
        plan = getattr(type(obj_or_cls), "__unpackable_plan__", None)
        if plan is None:
            plan = _plain_plan(
                obj_or_cls,
                fields=None,
                exclude=exclude,
                include_private=include_private,
                recursive=True,
                aliases=aliases,
            )
    if plan.static:
        return plan.public_names
    if isinstance(obj_or_cls, type):
        return ()
    return tuple(key for key, _ in _plan_items(obj_or_cls, plan))


def asdict(
    obj: Any,
    *,
    fields: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = (),
    include_private: bool = False,
    aliases: Optional[Mapping[str, str]] = None,
    recursive: Optional[bool] = None,
) -> Dict[str, Any]:
    has_overrides = bool(fields or exclude or include_private or aliases)
    if is_unpackable(obj) and not has_overrides:
        return obj.to_dict(recursive=recursive)

    should_recurse = True if recursive is None else recursive

    if isinstance(obj, dict):
        return {
            key: _convert_field(value, jsonable=False, recursive=should_recurse)
            for key, value in obj.items()
        }
    if is_unpackable(obj) or _is_plain_projectable(obj):
        return _project_dict(
            obj,
            fields=fields,
            exclude=exclude,
            include_private=include_private,
            aliases=aliases,
            recursive=should_recurse,
            jsonable=False,
        )
    raise TypeError(f"{type(obj).__name__} is not unpackable")


def astuple(
    obj: Any,
    *,
    fields: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = (),
    include_private: bool = False,
    aliases: Optional[Mapping[str, str]] = None,
    recursive: Optional[bool] = None,
) -> Tuple[Any, ...]:
    has_overrides = bool(fields or exclude or include_private or aliases)
    if is_unpackable(obj) and not has_overrides:
        return obj.to_tuple(recursive=recursive)
    if is_unpackable(obj) or _is_plain_projectable(obj):
        return _project_tuple(
            obj,
            fields=fields,
            exclude=exclude,
            include_private=include_private,
            aliases=aliases,
            recursive=True if recursive is None else recursive,
        )
    raise TypeError(f"{type(obj).__name__} is not unpackable")


def to_jsonable(obj: Any, *, recursive: bool = True) -> Any:
    return _convert(obj, jsonable=True, recursive=recursive)

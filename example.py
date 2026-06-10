from unpackable import asdict, unpackable


@unpackable
class Tick:
    __slots__ = ("symbol", "price", "size")

    def __init__(self, symbol: str, price: float, size: int) -> None:
        self.symbol = symbol
        self.price = price
        self.size = size


@unpackable(exclude={"_latency_ns"})
class Trade:
    __slots__ = ("tick", "venue", "_latency_ns")

    def __init__(self, tick: Tick, venue: str, latency_ns: int) -> None:
        self.tick = tick
        self.venue = venue
        self._latency_ns = latency_ns


tick = Tick("AAPL", 192.4, 100)
symbol, price, size = tick

trade = Trade(tick, "NASDAQ", 870)

print(symbol, price, size)
print({**tick})
print(trade.to_tuple())
print(asdict(trade))

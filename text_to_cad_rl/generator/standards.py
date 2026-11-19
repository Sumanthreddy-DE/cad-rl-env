"""Train-only standards tables.

ISO 273 clearance holes (fine, medium, coarse), mm. Values as verified in SimReady's
SP1 Task 2 review (IS 1821:1987 = identical adoption of ISO 273:1979, p.6 table).
"""

from __future__ import annotations

ISO273: dict[str, tuple[float, float, float]] = {
    "M2.5": (2.7, 2.9, 3.1),
    "M3": (3.2, 3.4, 3.6),
    "M4": (4.3, 4.5, 4.8),
    "M5": (5.3, 5.5, 5.8),
    "M6": (6.4, 6.6, 7.0),
    "M8": (8.4, 9.0, 10.0),
}

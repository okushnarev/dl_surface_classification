from dataclasses import dataclass, fields
from typing import Any, Self


class Style:
    def __init__(self, base_style: dict[str, Any] = None):
        if base_style is None:
            base_style = {
                'font_name':  'Calibri',
                'font_size':  15,
                'valign':     'vcenter',
                'align':      'left',
                'num_format': '0.0000',
            }

        self.style = base_style

    def __add__(self, other: Self | dict[str, Any]) -> Self:
        if not isinstance(other, dict):
            other = other.style
        return Style(self.style | other)

    def __getitem__(self, key: str) -> Any:
        return self.style[key]

    def __copy__(self) -> Self:
        return Style(self.style.copy())

    def copy(self) -> Self:
        return self.__copy__()

    def set(self, **kwargs: Any):
        for key, value in kwargs.items():
            self.style[key] = value


@dataclass
class SheetStyles:
    base: Style
    header: Style = None
    link: Style = None
    better_stats: Style = None
    separator: Style = None

    def __post_init__(self):
        for field in fields(self):
            if field.name != 'base' and getattr(self, field.name) is None:
                setattr(self, field.name, self.base)

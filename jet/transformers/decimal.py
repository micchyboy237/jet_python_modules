from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Union


def to_number(
    value: Decimal,
    as_type: Literal["float", "int", "cents"] = "float",
    rounding: str = ROUND_HALF_UP,
) -> Union[float, int]:
    """
    Safely convert a Decimal to float, int (with rounding), or integer cents.

    Args:
        value: The Decimal to convert
        as_type: One of "float", "int", "cents"
            - "float":    returns float with full precision
            - "int":      returns rounded integer (default rounding=ROUND_HALF_UP)
            - "cents":    returns integer after multiplying by 100 (common for money)
        rounding: Rounding mode when as_type="int" (decimal rounding string)

    Returns:
        float or int depending on as_type

    Raises:
        ValueError: if invalid as_type or rounding mode
    """
    if not isinstance(value, Decimal):
        raise TypeError("Input must be a decimal.Decimal object")

    match as_type:
        case "float":
            return float(value)

        case "int":
            # quantize to integer with chosen rounding, then convert
            quantized = value.quantize(Decimal("1"), rounding=rounding)
            return int(quantized)

        case "cents":
            # Most common money conversion: 19.99 → 1999
            cents_decimal = value * Decimal("100")
            # Round to nearest integer cent (bankser's rounding is safer here)
            quantized_cents = cents_decimal.quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            return int(quantized_cents)

        case _:
            raise ValueError(
                f"Invalid as_type: {as_type!r}. Use one of: 'float', 'int', 'cents'"
            )

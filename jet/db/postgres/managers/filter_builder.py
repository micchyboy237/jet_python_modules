# jet_python_modules/jet/db/postgres/managers/filter_builder.py
import json
from typing import Any, Dict, List, Optional, Tuple

from psycopg import sql


class FilterBuilder:
    """
    Generic SQL WHERE clause builder.

    Usage Examples:
    - Standard Equality: {"code": "SSIS-123"}
    - SQL IN Clause: {"code": {"$in": ["A", "B"]}}
    - JSONB Containment: {"genres": {"$contains": ["Action"]}}
    - Greater Than: {"release_date": {"$gte": "2023-01-01"}}
    - Is Null: {"title": None}
    """

    @staticmethod
    def build(filters: Optional[Dict[str, Any]]) -> Tuple[sql.Composable, List[Any]]:
        if not filters:
            return sql.SQL(""), []

        conditions = []
        params = []

        for key, value in filters.items():
            identifier = sql.Identifier(key)

            # 1. Handle Null Checks
            if value is None:
                conditions.append(sql.SQL("{} IS NULL").format(identifier))
                continue

            # 2. Handle Operator Dicts (e.g., {"$in": [...]})
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "$in":
                        if isinstance(val, list) and val:
                            placeholders = sql.SQL(", ").join(
                                [sql.Placeholder()] * len(val)
                            )
                            conditions.append(
                                sql.SQL("{} IN ({})").format(identifier, placeholders)
                            )
                            params.extend(val)
                    elif op == "$contains":
                        # Generic JSONB containment using @>
                        if val is not None:
                            conditions.append(
                                sql.SQL("{} @> %s::jsonb").format(identifier)
                            )
                            params.append(
                                json.dumps(val)
                                if isinstance(val, (list, dict))
                                else str(val)
                            )
                    elif op == "$gte":
                        conditions.append(sql.SQL("{} >= %s").format(identifier))
                        params.append(val)
                    elif op == "$lte":
                        conditions.append(sql.SQL("{} <= %s").format(identifier))
                        params.append(val)
                    elif op == "$gt":
                        conditions.append(sql.SQL("{} > %s").format(identifier))
                        params.append(val)
                    elif op == "$lt":
                        conditions.append(sql.SQL("{} < %s").format(identifier))
                        params.append(val)
                continue

            # 3. Handle Bare Lists (Default to IN clause for generic behavior)
            if isinstance(value, list):
                if value:
                    placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(value))
                    conditions.append(
                        sql.SQL("{} IN ({})").format(identifier, placeholders)
                    )
                    params.extend(value)
            else:
                # 4. Standard Equality
                conditions.append(sql.SQL("{} = %s").format(identifier))
                params.append(value)

        if not conditions:
            return sql.SQL(""), []

        return sql.SQL(" WHERE {}").format(sql.SQL(" AND ").join(conditions)), params

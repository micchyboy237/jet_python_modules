from typing import TypedDict


class DB_Options(TypedDict, total=False):
    type: str
    host: str
    port: int
    name: str
    user: str
    password: str
    connect_name: str


class DB_Result(TypedDict, total=False):
    success: bool
    command: str
    message: str


class SQLResult(TypedDict, total=False):
    success: bool
    db_name: str
    command: str
    message: str
    results: list[DB_Result]

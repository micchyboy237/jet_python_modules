"""Common starting URLs with descriptions for demos"""

from typing import TypedDict, Literal

class StartConfig(TypedDict):
    url: str
    description: str

STARTING_POINTS: dict[Literal[
    "ecommerce_demo",
    "todo_app",
    "practice_form",
    "tables_and_dialogs",
    "complex_ui"
], StartConfig] = {
    "ecommerce_demo": {
        "url": "https://www.saucedemo.com/",
        "description": "Classic simple e-commerce demo — login, product list, cart, checkout"
    },
    "todo_app": {
        "url": "https://todomvc.com/examples/react/dist/",
        "description": "TodoMVC React - very good for form, list, filtering, state interaction"
    },
    "practice_form": {
        "url": "https://demoqa.com/automation-practice-form",
        "description": "Rich practice form — inputs, dropdowns, datepicker, file upload, radio, checkbox"
    },
    "tables_and_dialogs": {
        "url": "https://the-internet.herokuapp.com/",
        "description": "Many small demo pages — tables, dialogs, drag & drop, multiple windows, etc"
    },
    "complex_ui": {
        "url": "https://mui.com/material-ui/getting-started/templates/dashboard/",
        "description": "Modern Material UI dashboard — many interactive components"
    }
}

DEFAULT_START: Literal["ecommerce_demo"] = "ecommerce_demo"

# Python Design Patterns Demo

> **Summary:** A collection of industry-standard design patterns implemented in Python 3.12. Focuses on readability, testability, and modern syntax like generic classes and type aliases.

## 📂 Structure

- **creational/**: Patterns for object creation (Dependency Injection, Factory).
- **structural/**: Patterns for code organization (Adapter, Decorator).
- **behavioral/**: Patterns for communication and logic (Observer, Strategy).
- **resource/**: Patterns for lifecycle management (Context Manager).

## 💡 Why These Patterns?

| Pattern                  | Benefit                                                                 |
| :----------------------- | :---------------------------------------------------------------------- |
| **Dependency Injection** | Makes code easier to test by removing hidden dependencies.              |
| **Factory Method**       | Decouples object creation from business logic.                          |
| **Strategy**             | Allows swapping algorithms at runtime without complex `if/else` chains. |
| **Decorator**            | Adds functionality (logging, auth) without modifying original code.     |

## 🛠️ Getting Started

```bash
python creational/dependency_injection.py
python behavioral/strategy_pattern.py
```

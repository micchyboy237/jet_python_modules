# Event Emissions Demo

> **Summary:** A demonstration of modern Python event-driven patterns including async buses, decorators, and Pydantic validation. Covers industry use cases for microservices, distributed systems, and audit trails with actionable recommendations.

This repository demonstrates modern event-driven design patterns in Python, focusing on industry standards for scalability and maintainability.

## 📂 Structure

- **patterns/**: Core implementations of different event bus architectures.
- **use_cases/**: Real-world examples (E-commerce, Analytics).
- **utils/**: Helper classes for logging and error handling.

## 🚀 Industry Standard Use Cases

### 1. Decoupled Microservices

Using an **Async Event Bus** to separate business logic from side effects (emails, notifications).

- _Why:_ Allows the core API to respond quickly while background tasks run concurrently.

### 2. Distributed Systems

Using **Redis Pub/Sub** or **Kafka** to communicate between services running on different servers.

- _Why:_ Ensures reliability and allows independent scaling of producers and consumers.

### 3. Audit Trails & History

Using **Event Sourcing** to store state changes as a sequence of immutable events.

- _Why:_ Provides a complete history of "what happened" and allows state reconstruction at any point in time.

## 💡 Recommendations

| Pattern                      | Best For                               | Tooling                |
| :--------------------------- | :------------------------------------- | :--------------------- |
| **In-Memory Async**          | Single-process apps, modular monoliths | `asyncio`, Custom Bus  |
| **Threaded Fire-and-Forget** | I/O-heavy sync apps (Flask/Django)     | `ThreadPoolExecutor`   |
| **Message Broker**           | Multi-service/distributed systems      | Redis, RabbitMQ, Kafka |
| **Event Sourcing**           | Financial systems, complex workflows   | Custom Aggregates + DB |

## 🛠️ Getting Started

```bash
pip install redis pydantic
python use_cases/ecommerce_demo.py
```

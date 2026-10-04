from opentelemetry.semconv.resource import ResourceAttributes

print("=== ResourceAttributes ===")
for name, value in vars(ResourceAttributes).items():
    if not name.startswith("_") and isinstance(value, str):
        print(f"{name} = {value}")

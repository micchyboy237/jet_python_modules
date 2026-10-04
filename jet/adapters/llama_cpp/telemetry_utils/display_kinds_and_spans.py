from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

print("=== OpenInferenceSpanKindValues ===")
for e in OpenInferenceSpanKindValues:
    print(f"{e.name} = {e.value}")

print()
print("=== SpanAttributes ===")
for name, value in vars(SpanAttributes).items():
    if not name.startswith("_") and isinstance(value, str):
        print(f"{name} = {value}")

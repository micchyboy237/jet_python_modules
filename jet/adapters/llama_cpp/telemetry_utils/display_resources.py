from opentelemetry.sdk.resources import Resource

# Create a resource
resource = Resource.create(
    {"service.name": "my-service", "deployment.environment": "production"}
)

# Method 1: Get all attributes as a dict
all_attrs = resource.attributes
print(all_attrs)
# Output: {'service.name': 'my-service', 'deployment.environment': 'production', ...}

# Method 2: Get a specific attribute
service_name = resource.attributes.get("service.name")
print(service_name)
# Output: 'my-service'

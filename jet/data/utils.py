import uuid


def generate_unique_hash(hash_length=24):
    # Generate a unique UUID and convert it to a string
    unique_hash = str(uuid.uuid4()).replace('-', '')  # Remove dashes
    # Return the hash truncated to the specified length
    return unique_hash[:hash_length]

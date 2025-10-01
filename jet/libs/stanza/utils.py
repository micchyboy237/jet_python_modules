from jet.utils.class_utils import get_non_empty_primitive_attributes, get_non_empty_object_attributes
import stanza

def serialize_stanza_object(data: stanza.models.common.stanza_object.StanzaObject) -> dict:
    """
    Recursively serialize stanza object
    """
    def _serialize(obj)-> dict[dict, any]:
        if isinstance(obj, dict):
            return {
                key: _serialize(value)
                for key, value in obj.items()
                if value is not None
                and key not in ["doc", "sent"]
            }
        elif isinstance(obj, list):
            return [_serialize(item) for item in obj]
        elif isinstance(obj, object):
            return get_non_empty_primitive_attributes(obj)
    serialized_data = {}
    for key, value in get_non_empty_object_attributes(data).items():
        if key not in ["ents"]:
            serialized_data[key] = _serialize(value)
    return serialized_data
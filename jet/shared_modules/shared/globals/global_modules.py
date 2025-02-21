import builtins

from .modules import (
    logger,

    make_serializable,
    format_json,

    class_to_string,
    validate_class,
    get_class_name,
    validate_iterable_class,
    get_iterable_class_name,
    get_non_empty_attributes,
    get_internal_attributes,
    get_callable_attributes,

    copy_to_clipboard,

    check_object_type,
    print_types_recursive,
    get_values_by_paths,
    extract_values_by_paths,

    get_max_prompt_char_length,
    clean_tags,
    clean_text,
    clean_newlines,
    clean_non_ascii,
    clean_other_characters,
    extract_sentences,
    extract_paragraphs,
    extract_sections,
    merge_texts,
    merge_texts_with_overlap,
    split_text,
)
# from jet.utils.class_utils import get_internal_attributes, get_non_empty_attributes, validate_class


# Injects global methods/variables only once
def inject_globals():
    if not hasattr(builtins, "logger"):
        builtins.logger = logger
    if not hasattr(builtins, "make_serializable"):
        builtins.make_serializable = make_serializable
    if not hasattr(builtins, "format_json"):
        builtins.format_json = format_json
    if not hasattr(builtins, "class_to_string"):
        builtins.class_to_string = class_to_string
    if not hasattr(builtins, "validate_class"):
        builtins.validate_class = validate_class
    if not hasattr(builtins, "get_class_name"):
        builtins.get_class_name = get_class_name
    if not hasattr(builtins, "validate_iterable_class"):
        builtins.validate_iterable_class = validate_iterable_class
    if not hasattr(builtins, "get_iterable_class_name"):
        builtins.get_iterable_class_name = get_iterable_class_name
    if not hasattr(builtins, "get_non_empty_attributes"):
        builtins.get_non_empty_attributes = get_non_empty_attributes
    if not hasattr(builtins, "get_internal_attributes"):
        builtins.get_internal_attributes = get_internal_attributes
    if not hasattr(builtins, "get_callable_attributes"):
        builtins.get_callable_attributes = get_callable_attributes
    if not hasattr(builtins, "copy_to_clipboard"):
        builtins.copy_to_clipboard = copy_to_clipboard
    if not hasattr(builtins, "check_object_type"):
        builtins.check_object_type = check_object_type
    if not hasattr(builtins, "print_types_recursive"):
        builtins.print_types_recursive = print_types_recursive
    if not hasattr(builtins, "get_values_by_paths"):
        builtins.get_values_by_paths = get_values_by_paths
    if not hasattr(builtins, "extract_values_by_paths"):
        builtins.extract_values_by_paths = extract_values_by_paths
    if not hasattr(builtins, "get_max_prompt_char_length"):
        builtins.get_max_prompt_char_length = get_max_prompt_char_length
    if not hasattr(builtins, "clean_tags"):
        builtins.clean_tags = clean_tags
    if not hasattr(builtins, "clean_text"):
        builtins.clean_text = clean_text
    if not hasattr(builtins, "clean_newlines"):
        builtins.clean_newlines = clean_newlines
    if not hasattr(builtins, "clean_non_ascii"):
        builtins.clean_non_ascii = clean_non_ascii
    if not hasattr(builtins, "clean_other_characters"):
        builtins.clean_other_characters = clean_other_characters
    if not hasattr(builtins, "extract_sentences"):
        builtins.extract_sentences = extract_sentences
    if not hasattr(builtins, "extract_paragraphs"):
        builtins.extract_paragraphs = extract_paragraphs
    if not hasattr(builtins, "extract_sections"):
        builtins.extract_sections = extract_sections
    if not hasattr(builtins, "merge_texts"):
        builtins.merge_texts = merge_texts
    if not hasattr(builtins, "merge_texts_with_overlap"):
        builtins.merge_texts_with_overlap = merge_texts_with_overlap
    if not hasattr(builtins, "split_text"):
        builtins.split_text = split_text


inject_globals()

__all__ = []

import builtins

from .import_modules import (
    logger,

    make_serializable,
    format_json,

    inspect_original_script_path,
    print_inspect_original_script_path,
    print_inspect_original_script_path_grouped,
    get_stack_frames,
    find_stack_frames,
    get_current_running_function,

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

    convert_html_to_markdown,
    html_to_markdown,
    scrape_markdown,

    get_flat_header_list,
    get_header_level,
    build_nodes_hierarchy,
    collect_nodes_full_content,
    get_header_contents,
    get_md_header_contents,
    merge_md_header_contents,
    extract_md_header_contents,
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
    if not hasattr(builtins, "inspect_original_script_path"):
        builtins.inspect_original_script_path = inspect_original_script_path
    if not hasattr(builtins, "print_inspect_original_script_path"):
        builtins.print_inspect_original_script_path = print_inspect_original_script_path
    if not hasattr(builtins, "print_inspect_original_script_path_grouped"):
        builtins.print_inspect_original_script_path_grouped = print_inspect_original_script_path_grouped
    if not hasattr(builtins, "get_stack_frames"):
        builtins.get_stack_frames = get_stack_frames
    if not hasattr(builtins, "find_stack_frames"):
        builtins.find_stack_frames = find_stack_frames
    if not hasattr(builtins, "get_current_running_function"):
        builtins.get_current_running_function = get_current_running_function
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
    if not hasattr(builtins, "convert_html_to_markdown"):
        builtins.convert_html_to_markdown = convert_html_to_markdown
    if not hasattr(builtins, "html_to_markdown"):
        builtins.html_to_markdown = html_to_markdown
    if not hasattr(builtins, "scrape_markdown"):
        builtins.scrape_markdown = scrape_markdown
    if not hasattr(builtins, "get_flat_header_list"):
        builtins.get_flat_header_list = get_flat_header_list
    if not hasattr(builtins, "get_header_level"):
        builtins.get_header_level = get_header_level
    if not hasattr(builtins, "build_nodes_hierarchy"):
        builtins.build_nodes_hierarchy = build_nodes_hierarchy
    if not hasattr(builtins, "collect_nodes_full_content"):
        builtins.collect_nodes_full_content = collect_nodes_full_content
    if not hasattr(builtins, "get_header_contents"):
        builtins.get_header_contents = get_header_contents
    if not hasattr(builtins, "get_md_header_contents"):
        builtins.get_md_header_contents = get_md_header_contents
    if not hasattr(builtins, "merge_md_header_contents"):
        builtins.merge_md_header_contents = merge_md_header_contents
    if not hasattr(builtins, "extract_md_header_contents"):
        builtins.extract_md_header_contents = extract_md_header_contents


inject_globals()

__all__ = []

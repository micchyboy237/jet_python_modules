import builtins

from .import_modules import (
    fix_and_unidecode,

    logger,

    make_serializable,
    format_json,

    inspect_original_script_path,
    print_inspect_original_script_path,
    print_inspect_original_script_path_grouped,
    get_stack_frames,
    find_stack_frames,
    get_current_running_function,

    is_class_instance,
    is_dictionary,
    class_to_string,
    validate_class,
    get_class_name,
    validate_iterable_class,
    get_iterable_class_name,
    get_builtin_attributes,
    get_non_empty_attributes,
    get_internal_attributes,
    get_callable_attributes,
    get_non_callable_attributes,

    copy_to_clipboard,

    check_object_type,
    print_types_recursive,
    get_values_by_paths,
    extract_values_by_paths,
    extract_null_keys,

    get_max_prompt_char_length,
    clean_tags,
    clean_text,
    clean_spaces,
    clean_newlines,
    clean_non_ascii,
    clean_other_characters,
    extract_sentences,
    extract_paragraphs,
    extract_sections,
    merge_texts,
    merge_texts_with_overlap,
    split_text,
    find_elements_with_text,
    extract_text_elements,
    extract_tree_with_text,
    format_html,
    print_html,

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

    save_cache,
    load_cache,
    load_or_save_cache,
    load_from_cache_or_compute,

    setup_nlp,
    pos_tag_nltk,
    pos_tag_spacy,
    split_words,
    get_words,
    get_non_words,
    count_words,
    count_non_words,
    process_dataset,
    process_all_datasets,
    compare_words,
    count_syllables,
    split_by_syllables,
    get_named_words,
    SpacyWord,
    get_spacy_words,
    list_all_spacy_pos_tags,

    process_sentence_newlines,
    handle_long_sentence,
    get_list_marker_pattern_substring,
    get_list_marker_pattern,
    get_list_sentence_pattern,
    is_ordered_list_marker,
    is_ordered_list_sentence,
    adaptive_split,
    split_sentences,
    merge_sentences,
    group_sentences,
    count_sentences,
    get_sentences,
    split_by_punctuations,
    encode_text_to_strings,
)
# from jet.utils.class_utils import get_internal_attributes, get_non_empty_attributes, validate_class


# Injects global methods/variables only once
def inject_globals():
    if not hasattr(builtins, "fix_and_unidecode"):
        builtins.fix_and_unidecode = fix_and_unidecode
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
    if not hasattr(builtins, "is_class_instance"):
        builtins.is_class_instance = is_class_instance
    if not hasattr(builtins, "is_dictionary"):
        builtins.is_dictionary = is_dictionary
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
    if not hasattr(builtins, "get_builtin_attributes"):
        builtins.get_builtin_attributes = get_builtin_attributes
    if not hasattr(builtins, "get_non_empty_attributes"):
        builtins.get_non_empty_attributes = get_non_empty_attributes
    if not hasattr(builtins, "get_internal_attributes"):
        builtins.get_internal_attributes = get_internal_attributes
    if not hasattr(builtins, "get_callable_attributes"):
        builtins.get_callable_attributes = get_callable_attributes
    if not hasattr(builtins, "get_non_callable_attributes"):
        builtins.get_non_callable_attributes = get_non_callable_attributes
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
    if not hasattr(builtins, "extract_null_keys"):
        builtins.extract_null_keys = extract_null_keys
    if not hasattr(builtins, "get_max_prompt_char_length"):
        builtins.get_max_prompt_char_length = get_max_prompt_char_length
    if not hasattr(builtins, "clean_tags"):
        builtins.clean_tags = clean_tags
    if not hasattr(builtins, "clean_text"):
        builtins.clean_text = clean_text
    if not hasattr(builtins, "clean_spaces"):
        builtins.clean_spaces = clean_spaces
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
    if not hasattr(builtins, "find_elements_with_text"):
        builtins.find_elements_with_text = find_elements_with_text
    if not hasattr(builtins, "extract_text_elements"):
        builtins.extract_text_elements = extract_text_elements
    if not hasattr(builtins, "extract_tree_with_text"):
        builtins.extract_tree_with_text = extract_tree_with_text
    if not hasattr(builtins, "format_html"):
        builtins.format_html = format_html
    if not hasattr(builtins, "print_html"):
        builtins.print_html = print_html
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
    if not hasattr(builtins, "save_cache"):
        builtins.save_cache = save_cache
    if not hasattr(builtins, "load_cache"):
        builtins.load_cache = load_cache
    if not hasattr(builtins, "load_or_save_cache"):
        builtins.load_or_save_cache = load_or_save_cache
    if not hasattr(builtins, "load_from_cache_or_compute"):
        builtins.load_from_cache_or_compute = load_from_cache_or_compute
    if not hasattr(builtins, "setup_nlp"):
        builtins.setup_nlp = setup_nlp
    if not hasattr(builtins, "pos_tag_nltk"):
        builtins.pos_tag_nltk = pos_tag_nltk
    if not hasattr(builtins, "pos_tag_spacy"):
        builtins.pos_tag_spacy = pos_tag_spacy
    if not hasattr(builtins, "split_words"):
        builtins.split_words = split_words
    if not hasattr(builtins, "get_words"):
        builtins.get_words = get_words
    if not hasattr(builtins, "get_non_words"):
        builtins.get_non_words = get_non_words
    if not hasattr(builtins, "count_words"):
        builtins.count_words = count_words
    if not hasattr(builtins, "count_non_words"):
        builtins.count_non_words = count_non_words
    if not hasattr(builtins, "process_dataset"):
        builtins.process_dataset = process_dataset
    if not hasattr(builtins, "process_all_datasets"):
        builtins.process_all_datasets = process_all_datasets
    if not hasattr(builtins, "compare_words"):
        builtins.compare_words = compare_words
    if not hasattr(builtins, "count_syllables"):
        builtins.count_syllables = count_syllables
    if not hasattr(builtins, "split_by_syllables"):
        builtins.split_by_syllables = split_by_syllables
    if not hasattr(builtins, "get_named_words"):
        builtins.get_named_words = get_named_words
    if not hasattr(builtins, "SpacyWord"):
        builtins.SpacyWord = SpacyWord
    if not hasattr(builtins, "get_spacy_words"):
        builtins.get_spacy_words = get_spacy_words
    if not hasattr(builtins, "list_all_spacy_pos_tags"):
        builtins.list_all_spacy_pos_tags = list_all_spacy_pos_tags
    if not hasattr(builtins, "process_sentence_newlines"):
        builtins.process_sentence_newlines = process_sentence_newlines
    if not hasattr(builtins, "handle_long_sentence"):
        builtins.handle_long_sentence = handle_long_sentence
    if not hasattr(builtins, "get_list_marker_pattern_substring"):
        builtins.get_list_marker_pattern_substring = get_list_marker_pattern_substring
    if not hasattr(builtins, "get_list_marker_pattern"):
        builtins.get_list_marker_pattern = get_list_marker_pattern
    if not hasattr(builtins, "get_list_sentence_pattern"):
        builtins.get_list_sentence_pattern = get_list_sentence_pattern
    if not hasattr(builtins, "is_ordered_list_marker"):
        builtins.is_ordered_list_marker = is_ordered_list_marker
    if not hasattr(builtins, "is_ordered_list_sentence"):
        builtins.is_ordered_list_sentence = is_ordered_list_sentence
    if not hasattr(builtins, "adaptive_split"):
        builtins.adaptive_split = adaptive_split
    if not hasattr(builtins, "split_sentences"):
        builtins.split_sentences = split_sentences
    if not hasattr(builtins, "merge_sentences"):
        builtins.merge_sentences = merge_sentences
    if not hasattr(builtins, "group_sentences"):
        builtins.group_sentences = group_sentences
    if not hasattr(builtins, "count_sentences"):
        builtins.count_sentences = count_sentences
    if not hasattr(builtins, "get_sentences"):
        builtins.get_sentences = get_sentences
    if not hasattr(builtins, "split_by_punctuations"):
        builtins.split_by_punctuations = split_by_punctuations
    if not hasattr(builtins, "encode_text_to_strings"):
        builtins.encode_text_to_strings = encode_text_to_strings


inject_globals()

__all__ = []

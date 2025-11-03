# List of HTML elements that inherently contain and render text visibly on a webpage
# Includes inline, block-level, list, table, form, and other text-rendering elements
# Excludes generic containers (e.g., div, section), void elements (e.g., img, br), and non-rendered text elements (e.g., script, style)
TEXT_ELEMENTS = [
    "a",        # Anchor for hyperlinks
    "abbr",     # Abbreviation or acronym
    "b",        # Bold text for stylistic emphasis
    "bdi",      # Bi-directional isolate
    "bdo",      # Bi-directional override
    "cite",     # Citation or reference
    "code",     # Inline code snippet
    "data",     # Machine-readable data
    "dfn",      # Definition term
    "em",       # Emphasized text
    "i",        # Italic text for styling
    "kbd",      # Keyboard input
    "mark",     # Highlighted text
    "q",        # Inline quotation
    "s",        # Strikethrough text
    "samp",     # Sample output
    "small",    # Small print for disclaimers
    "span",     # Generic inline container
    "strong",   # Strong importance
    "sub",      # Subscript text
    "sup",      # Superscript text
    "time",     # Time or date
    "u",        # Underlined text
    "var",      # Variable in math/programming
    "h1",       # Heading level 1
    "h2",       # Heading level 2
    "h3",       # Heading level 3
    "h4",       # Heading level 4
    "h5",       # Heading level 5
    "h6",       # Heading level 6
    "p",        # Paragraph
    "blockquote",  # Block quotation
    "pre",      # Preformatted text
    "address",  # Contact information
    "li",       # List item
    "dt",       # Definition term
    "dd",       # Definition description
    "th",       # Table header
    "td",       # Table data
    "caption",  # Table caption
    "label",    # Form control label
    "option",   # Select dropdown option
    "optgroup",  # Option group
    "legend",   # Fieldset caption
    "title",    # Document title
    "figcaption",  # Figure caption
    "summary",  # Summary for details element
    "ruby",     # Ruby annotation
    "rt",       # Ruby text
    "rp",       # Ruby parenthesis
    "del",      # Deleted text
    "ins"       # Inserted text
]

# Lists of HTML elements that inherently contain and render text visibly on a webpage
# Split into inline and block elements
# Excludes generic containers (e.g., div, section), void elements (e.g., img, br), and non-rendered text elements (e.g., script, style)

# Inline text elements: Render text within the flow of a line, typically inside block elements
INLINE_TEXT_ELEMENTS = [
    "a",        # Anchor for hyperlinks
    "abbr",     # Abbreviation or acronym
    "b",        # Bold text for stylistic emphasis
    "bdi",      # Bi-directional isolate
    "bdo",      # Bi-directional override
    "cite",     # Citation or reference
    "code",     # Inline code snippet
    "data",     # Machine-readable data
    "dfn",      # Definition term
    "em",       # Emphasized text
    "i",        # Italic text for styling
    "kbd",      # Keyboard input
    "mark",     # Highlighted text
    "q",        # Inline quotation
    "s",        # Strikethrough text
    "samp",     # Sample output
    "small",    # Small print for disclaimers
    "span",     # Generic inline container
    "strong",   # Strong importance
    "sub",      # Subscript text
    "sup",      # Superscript text
    "time",     # Time or date
    "u",        # Underlined text
    "var",      # Variable in math/programming
    "del",      # Deleted text
    "ins",      # Inserted text
    "rt",       # Ruby text
    "rp",       # Ruby parenthesis
]

# Block text elements: Render text as a block, typically starting on a new line
BLOCK_TEXT_ELEMENTS = [
    "h1",       # Heading level 1
    "h2",       # Heading level 2
    "h3",       # Heading level 3
    "h4",       # Heading level 4
    "h5",       # Heading level 5
    "h6",       # Heading level 6
    "p",        # Paragraph
    "blockquote",  # Block quotation
    "pre",      # Preformatted text
    "address",  # Contact information
    "li",       # List item
    "dt",       # Definition term
    "dd",       # Definition description
    "th",       # Table header
    "td",       # Table data
    "caption",  # Table caption
    "label",    # Form control label
    "option",   # Select dropdown option
    "optgroup",  # Option group
    "legend",   # Fieldset caption
    "title",    # Document title
    "figcaption",  # Figure caption
    "summary",  # Summary for details element
    "ruby",     # Ruby annotation
]

TEXT_ELEMENTS = INLINE_TEXT_ELEMENTS + BLOCK_TEXT_ELEMENTS

HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]

JS_UTILS_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/scrapers/browser/scripts/utils.js"

PROMPT_DICT = {
    # Prompt templates used for document layout detection
    "layout_detect": (
        "Analyze the layout structure of the input document, detect all structural elements "
        "and classify them semantically. Use \n to delimit different regions."
    ),

    # Template for recognizing text within a bounding box.
    "text_recognize": (
        "Based on the given input field coordinates and layout type, identify and extract "
        "the content within the specified region. Formulas shall be represented in LaTeX "
        "notation, and tables shall be structured in OTSL format: "
        "<x_{}><y_{}><x_{}><y_{}><LAYOUT_TEXT>"
    ),

    # Template for recognizing title within a bounding box.
    "title_recognize": (
        "Based on the given input field coordinates and layout type, identify and extract "
        "the content within the specified region. Formulas shall be represented in LaTeX "
        "notation, and tables shall be structured in OTSL format: "
        "<x_{}><y_{}><x_{}><y_{}><LAYOUT_TITLE>"
    ),

    # Template for recognizing table within a bounding box.
    "table_recognize": (
        "Based on the given input field coordinates and layout type, identify and extract "
        "the content within the specified region. Formulas shall be represented in LaTeX "
        "notation, and tables shall be structured in OTSL format: "
        "<x_{}><y_{}><x_{}><y_{}><LAYOUT_TABLE>"
    ),

    # Template for recognizing chart within a bounding box.
    "chart_data_recognize": (
        "Based on the given input field coordinates and layout type, identify and extract "
        "the content within the specified region. Formulas shall be represented in LaTeX "
        "notation, and tables shall be structured in OTSL format: "
        "<x_{}><y_{}><x_{}><y_{}><LAYOUT_CHART_DATA>"
    ),

    "chart_logic_recognize": (
        "Based on the given input field coordinates and layout type, identify and extract "
        "the content within the specified region. Formulas shall be represented in LaTeX "
        "notation, and tables shall be structured in OTSL format: "
        "<x_{}><y_{}><x_{}><y_{}><LAYOUT_CHART_LOGIC>"
    ),

    # "chart_recognize": (
    #     "Convert the logic charts in the figure to Mermaid format and the data charts "
    #     "to Markdown format."
    # ),

    # Template for recognizing seal within a bounding box.
    "seal_recognize": (
        "Based on the given input field coordinates and layout type, identify and extract "
        "the content within the specified region. Formulas shall be represented in LaTeX "
        "notation, and tables shall be structured in OTSL format: "
        "<x_{}><y_{}><x_{}><y_{}><LAYOUT_SEAL>"
    ),

    # Template for recognizing figure content
    "figure_recognize": (
        "Extract all textual elements from the given figure and present the recognition "
        "results in a structured manner."
    ),

    "hierarchy_recognize": (
        "Identify hierarchical relationships among input fields and output the hierarchy structure."
    )
}


import re

def parse_ostl_row_str(row_str):
    """
    Parse a OSTL row string and extract each keyword and its corresponding content.

    Args:
        row_str (str): The input OSTL row string.

    Returns:
        List[List[str, str]]: A list of [keyword, content] pairs.
    """
    ostl_keywords = ['<start>', '<left>', '<up>', '<left_up>']
    keyword_pattern = '|'.join(re.escape(k) for k in ostl_keywords)
    
    # Collect each keyword and its start position in the string
    keyword_positions = [
        (match.group(), match.start())
        for match in re.finditer(keyword_pattern, row_str)
    ]

    result_cells = []
    # Iterate through each keyword and slice out the content region
    for idx, (keyword, pos) in enumerate(keyword_positions):
        start_idx = pos
        # Compute the content slicing interval
        end_idx = keyword_positions[idx + 1][1] if idx < len(keyword_positions) - 1 else len(row_str)

        cell_segment = row_str[start_idx:end_idx]
        # Extract content following <content>
        cell_content = cell_segment.split('<content>')[-1]

        result_cells.append([keyword, cell_content])

    return result_cells

def handle_start_cell(cell_structure, row_idx, col_idx, cell_id):
    cell_structure[row_idx].append(cell_id)

def handle_left_cell(cell_structure, row_idx, col_idx):
    if col_idx > 0:
        cell_structure[row_idx].append(cell_structure[row_idx][col_idx - 1])
    else:
        cell_structure[row_idx].append(-1)

def handle_up_cell(cell_structure, row_idx, col_idx):
    if row_idx > 0 and len(cell_structure[row_idx - 1]) > col_idx:
        cell_structure[row_idx].append(cell_structure[row_idx - 1][col_idx])
    else:
        cell_structure[row_idx].append(-1)

def handle_left_up_cell(cell_structure, row_idx, col_idx):
    if (row_idx > 0 and col_idx > 0 
            and len(cell_structure[row_idx - 1]) > col_idx 
            and cell_structure[row_idx - 1][col_idx] == cell_structure[row_idx][col_idx - 1]):
        cell_structure[row_idx].append(cell_structure[row_idx - 1][col_idx])
    else:
        cell_structure[row_idx].append(-1)

def parse_ostl_table_str(table_str):
    """
    Parse an OSTL format table string, return table structure and cell content mapping.

    Args:
        table_str (str): OSTL table string.

    Returns:
        tuple:
            cell_structure: Row-column cell ID matrix
            cell_content: Mapping of cell ID to content
    """

    # Split by lines to make a list
    rows = table_str.strip().splitlines()

    # Initialize structure and content
    cell_structure = [[] for _ in range(len(rows))]
    cell_content = {}

    cell_id = 0  # Current available cell ID

    for row_idx, row_str in enumerate(rows):
        cells = parse_ostl_row_str(row_str)
        for col_idx, (cell_type, cell_text) in enumerate(cells):
            if cell_type == '<start>':
                # Create new cell
                handle_start_cell(cell_structure, row_idx, col_idx, cell_id)
                cell_content[cell_id] = cell_text if isinstance(cell_text, str) else ''
                cell_id += 1
            elif cell_type == '<left>':
                # Reference left cell
                handle_left_cell(cell_structure, row_idx, col_idx)
            elif cell_type == '<up>':
                # Reference upper cell
                handle_up_cell(cell_structure, row_idx, col_idx)
            else:
                # Reference upper-left cell
                handle_left_up_cell(cell_structure, row_idx, col_idx)

    # Pad to N×M format, make each row the same length
    max_cols = max(len(row) for row in cell_structure)
    for row in cell_structure:
        row.extend([-1] * (max_cols - len(row)))

    # Set empty string for special ID -1
    cell_content[-1] = ''

    return cell_structure, cell_content

def get_colspan(cell_structure, row_idx, col_idx, cell_id):
    """Count how many columns this cell spans horizontally."""
    next_col = col_idx + 1
    row = cell_structure[row_idx]
    while next_col < len(row) and row[next_col] == cell_id:
        next_col += 1
    return next_col - col_idx

def get_rowspan(cell_structure, row_idx, col_idx, cell_id):
    """Count how many rows this cell spans vertically."""
    next_row = row_idx + 1
    while next_row < len(cell_structure) and cell_structure[next_row][col_idx] == cell_id:
        next_row += 1
    return next_row - row_idx

def make_td(cell_id, cell_text, rowspan, colspan):
    """Compose the <td> string for this cell."""
    attrs = []
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    attr_str = (' ' + ' '.join(attrs)) if attrs else ''
    return f'<td{attr_str}>{cell_text}</td>'

def pack_to_html_str(cell_structure, cell_content):
    """
    Convert cell structure and content mapping into an HTML table string.

    Args:
        cell_structure (List[List[int]]): Table cell ID structure matrix.
        cell_content (Dict[int, str]): Cell ID to text content mapping.

    Returns:
        str: HTML table string.
    """
    used_cell_ids = set()
    html_parts = ['<table>']

    for row_idx, row in enumerate(cell_structure):
        html_parts.append('<tr>')
        for col_idx, cell_id in enumerate(row):
            # Skip already rendered merged cells
            if cell_id in used_cell_ids:
                continue

            # Mark this cell ID as rendered (if valid)
            if cell_id != -1:
                used_cell_ids.add(cell_id)

            # Calculate colspan/rowspan
            colspan = get_colspan(cell_structure, row_idx, col_idx, cell_id)
            rowspan = get_rowspan(cell_structure, row_idx, col_idx, cell_id)

            # If invalid ID, span is always 1
            if cell_id == -1:
                colspan = 1
                rowspan = 1

            cell_text = cell_content.get(cell_id, '')

            td_html = make_td(cell_id, cell_text, rowspan, colspan)
            html_parts.append(td_html)
        html_parts.append('</tr>')
    html_parts.append('</table>')
    return ''.join(html_parts)

def convert_table_ostl_to_html(ostl_str):
    """
    Convert OSTL format table string to HTML format table string.

    Args:
        ostl_str (str): OSTL format table string.

    Returns:
        str: HTML format table string.
    """
    # Parse OSTL table to cell structure and content mapping
    cell_structure, cell_content = parse_ostl_table_str(ostl_str)
    # Assemble to HTML table
    html_str = pack_to_html_str(cell_structure, cell_content)
    return html_str

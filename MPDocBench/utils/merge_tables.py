import json
import os
from bs4 import BeautifulSoup
import html
import unicodedata
import re
import collections
from tqdm import tqdm


def full_to_half(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return "".join(result)

def normalized_html_table(text):
    def process_table_html(md_i):
        def process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()  # unwrap()会移除标签但保留其内容
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res=''
        if '<table' in md_i.replace(" ","").replace("'",'"'):
            md_i = process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            # table_res = re.sub('<table.*?>','',table_res)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>',"",table_res)
            table_res = '<table border="1" >' + table_res + '</table>'

        return table_res

    def extract_and_clean_tables(text):
        if '</table>' not in text:
            text += '</table>'
        tables = re.findall(r'<table.*?>.*?</table>', text, re.DOTALL)
        if len(tables) > 1:
            return ""
        clean_tables = []
        for table in tables:
            table_content = re.sub(r'<table.*?>', '<table>', table)
            table_content = re.sub(r'>\s+<', '><', table_content)
            table_content = re.sub(r'<td(.*?)>(.*?)</td>', lambda m: '<td' + m.group(1) + ">" + m.group(2).strip() + '</td>', table_content, flags=re.DOTALL)
            table_content = re.sub(r'>(.*?)<', lambda m: '>' + m.group(1).replace('\n', '').strip() + '<', table_content, flags=re.DOTALL)
            table_content = table_content.replace('\n', '').strip()
            clean_tables.append(table_content)
        flat_table = ''.join(clean_tables)
        return flat_table
    
    def clean_table(input_str,flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>','',input_str)
        return input_str
    
    norm_text = process_table_html(text)
    norm_text = clean_table(extract_and_clean_tables(norm_text))
    return norm_text


def merge_texts(content_lists):
    return "".join(content_lists)



def detect_table_headers(soup1, soup2, max_header_rows=5):
    """
    Determine how many identical rows exist at the beginning of two tables
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    # Check only the minimum number of rows
    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    for i in range(min_rows):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])
        if len(cells1) != len(cells2):
            headers_match = header_rows > 0
            break
        # If column counts match, check if content is identical
        match = True
        for c1, c2 in zip(cells1, cells2):
            text1 = "".join(full_to_half(c1.get_text()).split())
            text2 = "".join(full_to_half(c2.get_text()).split())
            if text1 != text2 or int(c1.get("colspan", 1)) != int(c2.get("colspan", 1)):
                match = False
                break
        # Complete match, increment matched row count. Otherwise, stop matching.
        if match:
            header_rows += 1
        else:
            headers_match = header_rows > 0
            break
    if header_rows == 0:
        headers_match = False
    return header_rows, headers_match


def calculate_table_total_columns(soup):
    """
    calculate total columns including colspan and rowspan, accounting for merged cells
    """
    rows = soup.find_all("tr")
    if not rows:
        return 0
    max_cols = 0
    occupied = {}
    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])
        if row_idx not in occupied:
            occupied[row_idx] = {}
        for cell in cells:
            while col_idx in occupied[row_idx]:
                col_idx += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True
            col_idx += colspan
            max_cols = max(max_cols, col_idx)
    return max_cols

def calculate_row_columns(row):
    """
    Calculate the actual number of columns in a single row
    """
    return sum(int(cell.get("colspan", 1)) for cell in row.find_all(["td", "th"]))


def calculate_visual_columns(row):
    """
    Calculate the visual number of columns in a single row, excluding colspan (merged cells count as one)
    """
    return len(row.find_all(["td", "th"]))

def check_rows_match(soup1, soup2):
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    if not rows1 or not rows2:
        return False
    last_row = rows1[-1]
    header_count, _ = detect_table_headers(soup1, soup2)
    first_data_row = rows2[header_count] if len(rows2) > header_count else None
    if not first_data_row:
        return False
    last_cols = calculate_row_columns(last_row)
    first_cols = calculate_row_columns(first_data_row)
    last_visual = calculate_visual_columns(last_row)
    first_visual = calculate_visual_columns(first_data_row)
    return last_cols == first_cols or last_visual == first_visual

def can_merge_tables(html_curr, html_prev):
    if not html_prev or not html_curr:
        return False, None, None
    soup_prev = BeautifulSoup(html_prev, "html.parser")
    soup_curr = BeautifulSoup(html_curr, "html.parser")

    total_cols_prev = calculate_table_total_columns(soup_prev)
    total_cols_curr = calculate_table_total_columns(soup_curr)
    tables_match = total_cols_prev == total_cols_curr
    rows_match = check_rows_match(soup_prev, soup_curr)

    return (tables_match or rows_match), soup_prev, soup_curr


def perform_table_merge(soup_prev, soup_curr):
    header_count, _ = detect_table_headers(soup_prev, soup_curr)
    rows_prev = soup_prev.find_all("tr")
    rows_curr = soup_curr.find_all("tr")
    for row in rows_curr[header_count:]:
        row.extract()
        rows_prev[-1].parent.append(row)
    return str(soup_prev)

def merge_tables(table_lists):
    if len(table_lists) == 1:
        return table_lists[0]
    elif len(table_lists) == 0:
        print("table_lists 的长度不能为0")
        return ""
    else:
        # print("table_lists 的长度为:", len(table_lists))
        curr_html = table_lists[-1]
        for i in range(len(table_lists)-2, -1, -1):
            prev_html = table_lists[i]
            can_merge, soup_prev, soup_curr = can_merge_tables(curr_html, prev_html)
            if can_merge:
                curr_html = perform_table_merge(soup_prev, soup_curr)
            else:
                print("some tables can not be merged")
        return curr_html
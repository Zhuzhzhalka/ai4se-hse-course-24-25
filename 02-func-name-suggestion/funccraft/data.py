import datasets
import pandas as pd
import tree_sitter_python as tspython
from pathlib import Path
from tree_sitter import Language, Parser


PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def print_example():
    example = """def sina_xml_to_url_list(xml_data):
        \"\"\"str->list
        Convert XML to URL List.
        From Biligrab.
        \"\"\"
        rawurl = []
        # Comment1
        # Comment 2
        dom = parseString(xml_data)
        for node in dom.getElementsByTagName('durl'):
            url = node.getElementsByTagName('url')[0]  # Comment 3
            rawurl.append(url.childNodes[0].data)
        return rawurl
    """
    print("Example")
    print(example)
    function_name, body_with_comments, body_without_comments = (
        extract_function_info(example)
    )
    print("Function name: ", function_name)
    print("Body with comments: \n", body_with_comments)
    print("Body without comments: \n", body_without_comments)


def pretty_print(fail):
    print_example()
    print("Number of failed func names:", fail)


def delete_symbols(delete_ranges, code):
    if not delete_ranges:
        return code

    parts = [code[0 : delete_ranges[0][0]]]
    for i in range(len(delete_ranges) - 1):
        parts.append(code[delete_ranges[i][1] : delete_ranges[i + 1][0]])
    parts.append(code[delete_ranges[-1][1] :])

    return ''.join(parts)


def delete_comments(node):
    comments = []

    for child in node.children:
        if child.type == 'comment':
            comments.append((child.start_byte, child.end_byte))
        elif (
            child.type == 'expression_statement'
            and child.children[0].type == 'string'
            and child.parent.parent.type == 'function_definition'
        ):
            comments.append((child.start_byte, child.end_byte))

        comments.extend(delete_comments(child))

    return comments


def extract_function_info(code):
    tree = parser.parse(bytes(code, "utf8"))
    function_name = None
    body_with_comments = ""
    body_without_comments = ""

    for child in tree.root_node.children:
        if child.type == "function_definition":
            function_name = child.child_by_field_name("name").text.decode(
                "utf-8"
            )
            body_with_comments = child.text.decode("utf-8")
            to_delete = delete_comments(child.child_by_field_name("body"))

            body_without_comments = (
                delete_symbols(to_delete, body_with_comments)
                if to_delete
                else body_with_comments
            )

            body_without_comments = '\n'.join(
                filter(None, map(str.strip, body_without_comments.split('\n')))
            )

            body_without_comments, body_with_comments = map(
                lambda x: x.replace(function_name, "<extra_id_0>", 1),
                [body_without_comments, body_with_comments],
            )

    return function_name, body_with_comments, body_without_comments


def check_correctness(dataset):
    n_fail = 0

    for i in range(1100):
        cur = dataset["func_name"][i].rsplit(".", 1)
        cur = cur[1] if len(cur) > 1 else cur[0]
        curr = dataset["custom_fn_name"][i]
        if cur != curr:
            n_fail += 1

    pretty_print(n_fail)


def prepare() -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "code_search_net", "python", split='test', trust_remote_code=True
    ).select(range(1100))
    dataset = pd.DataFrame(dataset)
    data = dataset['whole_func_string']
    func_names = []
    with_comments = []
    without_comments = []

    for i in data:
        function_name, body_with_comments, body_without_comments = (
            extract_function_info(i)
        )
        func_names.append(function_name)
        with_comments.append(body_with_comments)
        without_comments.append(body_without_comments)

    dataset['custom_fn_name'] = func_names
    dataset['custom_with_comments'] = with_comments
    dataset['custom_without_comments'] = without_comments

    check_correctness(dataset)

    return datasets.Dataset.from_pandas(dataset)


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))

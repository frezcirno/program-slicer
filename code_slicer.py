import shutil
import pydot
from collections import defaultdict
import os
from igraph import Graph, Vertex
from csv import DictReader
from tqdm import tqdm
import subprocess as sp
import Levenshtein
import re

DIR = os.path.dirname(os.path.abspath(__file__))
SENSI_API_LIST = os.path.join(DIR, "resources", "sensiAPI.txt")
with open(SENSI_API_LIST, "r", encoding="utf-8") as f:
    SENSI_API_SET = frozenset([api.strip() for api in f.read().split(",")])

# up to C11 and C++17; immutable set
CPPKEYWORDS = frozenset(
    {
        "__asm",
        "__builtin",
        "__cdecl",
        "__declspec",
        "__except",
        "__export",
        "__far16",
        "__far32",
        "__fastcall",
        "__finally",
        "__import",
        "__inline",
        "__int16",
        "__int32",
        "__int64",
        "__int8",
        "__leave",
        "__optlink",
        "__packed",
        "__pascal",
        "__stdcall",
        "__system",
        "__thread",
        "__try",
        "__unaligned",
        "_asm",
        "_Builtin",
        "_Cdecl",
        "_declspec",
        "_except",
        "_Export",
        "_Far16",
        "_Far32",
        "_Fastcall",
        "_finally",
        "_Import",
        "_inline",
        "_int16",
        "_int32",
        "_int64",
        "_int8",
        "_leave",
        "_Optlink",
        "_Packed",
        "_Pascal",
        "_stdcall",
        "_System",
        "_try",
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "const",
        "const_cast",
        "constexpr",
        "continue",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "final",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "override",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
        "NULL",
    }
)

# regular expression to find function name candidates
REGX_FUNCALL = re.compile(r"\b([_A-Za-z]\w*)\b(?=\s*\()")
# regular expression to find variable name candidates
# rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
REGX_VAR = re.compile(r"\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()")

MAIN = frozenset({"main"})
MAINARGS = frozenset({"argc", "argv"})
ALLKEYWORDS = CPPKEYWORDS.union(SENSI_API_SET)


def load_graph(node_file, edge_file, chunksiz=20000):
    graph = Graph(directed=True)

    with open(node_file) as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        chunk = []
        chunk_attr = defaultdict(list)
        for lineno, row in tqdm(enumerate(reader)):
            if "key" not in row:
                print("Bad row: ", row)
                continue
            key = row.pop("key")
            assert str(lineno) == key
            chunk.append(key)
            for key, val in row.items():
                chunk_attr[key].append(val)
            if len(chunk) >= chunksiz:
                graph.add_vertices(chunk, chunk_attr)
                chunk = []
                chunk_attr = defaultdict(list)
        if chunk:
            graph.add_vertices(chunk, chunk_attr)

    with open(edge_file) as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        chunk = []
        chunk_attr = defaultdict(list)
        for row in tqdm(reader):
            start = row.pop("start")
            end = row.pop("end")
            chunk.append((start, end))
            for key, val in row.items():
                chunk_attr[key].append(val)
            if len(chunk) >= chunksiz:
                graph.add_edges(chunk, chunk_attr)
                chunk = []
                chunk_attr = defaultdict(list)
        if chunk:
            graph.add_edges(chunk, chunk_attr)

    return graph


def plot_dot(cpg, filename="cpg.dot", render=False):
    dot = pydot.Graph()
    for v in cpg.vs:
        dot.add_node(pydot.Node(v.index, label=v["code"]))

    for e in cpg.es:
        etype = e["type"]
        color = "black"
        if e["type"] in {"IS_AST_PARENT"}:
            etype = ""
            color = "blue"
        elif e["type"] in {"IS_FUNCTION_OF_CFG", "IS_FUNCTION_OF_AST", "IS_FILE_OF"}:
            etype = ""
            color = "grey"
        elif e["type"] in {"CONTROLS", "CDG"}:
            etype = ""
            color = "red"
        elif e["type"] in {"DOM", "POST_DOM"}:
            # 和CONTROLS基本一致
            continue
            etype = ""
            color = "pink"
        elif e["type"] in {"FLOWS_TO"}:
            continue
        elif e["type"] in {"DEF", "USE"}:
            continue
        elif e["type"] in {"REACHES", "DDG"}:
            etype = ""
            color = "purple"
        else:
            color = "red"

        edge = pydot.Edge(e.source, e.target, label=etype)
        edge.set("color", color)
        dot.add_edge(edge)

    with open(filename, "w") as f:
        dotstr = dot.to_string()
        dotstr = dotstr.replace(r'\\"', r"\"")
        dotstr = dotstr.replace(r'\\"', r"\"")
        f.write(dotstr)

    if render:
        os.system(f"dot -Tpng {filename} -o {filename}.png")


def find_parent_by_etype(
    v: Vertex,
    etype={"IS_AST_PARENT", "IS_FUNCTION_OF_AST", "IS_FILE_OF", "IS_FUNCTION_OF_CFG"},
):
    if isinstance(etype, str):
        etype = frozenset([etype])
    for edge in v.in_edges():
        if edge["type"] in etype:
            return edge.source_vertex


def find_children_by_etype(
    v: Vertex,
    etype={"IS_AST_PARENT", "IS_FUNCTION_OF_AST", "IS_FILE_OF", "IS_FUNCTION_OF_CFG"},
):
    if isinstance(etype, str):
        etype = frozenset([etype])
    for edge in v.out_edges():
        if edge["type"] in etype:
            return edge.target_vertex


def find_children_in_vtype(v: Vertex, vtype):
    if isinstance(vtype, str):
        vtype = frozenset([vtype])
    for edge in v.out_edges():
        if edge.target_vertex["type"] in vtype:
            return edge.target_vertex


TOP_VTYPES = {
    "Function",
    "ExpressionStatement",
    "IdentifierDeclStatement",
    "IfStatement",
    "ElseStatement",
    "SwitchStatement",
    "Label",
    "ForStatement",
    "WhileStatement",
    "DoStatement",
    "BreakStatement",
    "ReturnStatement",
}


def find_top_node(v: Vertex):
    parent = v
    while parent:
        if parent["type"] in TOP_VTYPES:
            return parent
        parent = find_parent_by_etype(parent)


def find_function(v: Vertex):
    parent = v
    while parent:
        if parent["type"] == "Function":
            return parent
        parent = find_parent_by_etype(parent)


def find_file(v: Vertex):
    parent = v
    while parent:
        if parent["type"] == "File":
            return parent
        parent = find_parent_by_etype(parent)


def get_file(v: Vertex):
    parent = find_file(v)
    if parent:
        return parent["code"]


def get_function_sig(v: Vertex):
    defv = v.graph.vs[v.index + 1]
    retv = find_children_in_vtype(defv, "ReturnType")
    code = (retv["code"] + " " if retv else "") + defv["code"] + " {"
    line = int(v["location"].split(":", 1)[0])
    return line, code


def get_ctrl_sig(v: Vertex):
    condv = v.graph.vs[v.index + 1]
    line = int(condv["location"].split(":", 1)[0])
    code = v["code"] + " {"
    return line, code


def get_top_node_info(v: Vertex):
    """Get Statement Node for Element Node"""
    file = get_file(v)
    if file is None:
        return None, None

    if v["type"] in TOP_VTYPES:
        stmt = v
    else:
        stmt = find_top_node(v)
        if stmt is None:
            return None, None

    if stmt["type"] in {"Function"}:
        line, code = get_function_sig(stmt)
    elif stmt["type"] in {
        "IfStatement",
        "ForStatement",  # next: ForInit
        "SwitchStatement",
        "WhileStatement",
    }:
        line, code = get_ctrl_sig(stmt)
    elif stmt["type"] in {"DoStatement"}:
        nextv = v.graph.vs[stmt.index + 1]
        if nextv["type"] == "CompoundStatement":
            # joern bug: wrong location
            nextv = v.graph.vs[stmt.index + 2]
        line = int(nextv["location"].split(":", 1)[0]) - 1
        code = stmt["code"] + " {"
    elif stmt["type"] in {"ElseStatement"}:
        nextv = v.graph.vs[stmt.index + 1]
        nextvtype = nextv["type"]
        if nextvtype == "IfStatement":
            line, code = get_ctrl_sig(nextv)
        else:  # ExpressionStatement CompoundStatement
            line = int(nextv["location"].split(":", 1)[0])
            code = stmt["code"] + " ;"
    else:  # break; return; identifierdeclstatement ; ...
        line = int(stmt["location"].split(":", 1)[0])
        code = stmt["code"]
        if stmt["type"] in {"ExpressionStatement"}:
            code += " ;"
    key = code[:10]
    key = re.sub(r"[^A-Za-z0-9_]", "", key)
    code = code.strip()
    return f"{file}:{line}:{key}", {"code": code, "cpgidx": stmt.index}


def get_functions(cpg: Graph):
    funcs = {}
    for v in cpg.vs:
        if v["type"] == "CFGEntryNode":
            fnv = find_parent_by_etype(v, "IS_FUNCTION_OF_CFG")
            if fnv is None:
                continue
            filev = find_file(fnv)
            if filev is None:
                continue
            fname = fnv["code"]
            filename = filev["code"]
            funcs[(filename, fname)] = fnv.index
    return funcs


def add_call_edges(cpg: Graph, funcs: dict):
    for v in cpg.vs:
        if v["type"] == "CallExpression":
            calleev = v.graph.vs[v.index + 1]
            if calleev is None:
                continue
            fname = calleev["code"]
            filev = find_file(v)
            if filev is None:
                continue
            filename = filev["code"]

            matched_fns = [func for func in funcs if func[1] == fname]
            if not matched_fns:
                continue
            if len(matched_fns) > 1:
                # select the most similar one
                matched_fns = sorted(
                    matched_fns, key=lambda x: Levenshtein.distance(x[0], filename)
                )
            matched_fn = matched_fns[0]

            cpg.add_edge(v, funcs[matched_fn], type="CALLS")


def get_statement_code(cpg: Graph, location: str, basedir: str):
    file, ln, _ = location.split(":")
    lastdir = os.path.basename(basedir.rstrip("/"))
    file = file.split(lastdir + "/", 1)[1]
    file = os.path.join(basedir, file)
    cpg.vs.find()
    with open(file) as f:
        lines = f.readlines()
        return lines[int(ln) - 1].rstrip()


def merge_nodes(cpg: Graph):
    edges = {}
    for e in cpg.es:
        if e["type"] in {"CONTROLS", "CALLS"}:
            src_key, src_info = get_top_node_info(e.source_vertex)
            dst_key, dst_info = get_top_node_info(e.target_vertex)
            if not src_key or not dst_key or src_key == dst_key:
                continue
            edges[(src_key, dst_key, "CDG", e["var"])] = (src_info, dst_info)
        elif e["type"] == "REACHES":
            src_key, src_info = get_top_node_info(e.source_vertex)
            dst_key, dst_info = get_top_node_info(e.target_vertex)
            if not src_key or not dst_key or src_key == dst_key:
                continue
            edges[(src_key, dst_key, "DDG", e["var"])] = (src_info, dst_info)

    line_pdg = Graph(directed=True)
    vs = []
    vsset = set()
    vattrs = {"code": [], "cpgidx": []}
    es = []
    eattrs = {"type": [], "var": []}
    for src, dst, etype, var in edges:
        src_info, dst_info = edges[(src, dst, etype, var)]
        if src not in vsset:
            vs.append(src)
            vsset.add(src)
            vattrs["code"].append(src_info["code"])
            vattrs["cpgidx"].append(src_info["cpgidx"])
        if dst not in vsset:
            vs.append(dst)
            vsset.add(dst)
            vattrs["code"].append(dst_info["code"])
            vattrs["cpgidx"].append(dst_info["cpgidx"])
        es.append((src, dst))
        eattrs["type"].append(etype)
        eattrs["var"].append(var)

    line_pdg.add_vertices(vs, vattrs)
    line_pdg.add_edges(es, eattrs)

    return line_pdg


def slice_graph_by(v: Vertex) -> list[Vertex]:
    vset = set()

    stack = [v]
    while stack:
        curv = stack.pop()
        vset.add(curv)
        for ine in curv.in_edges():
            if ine.source_vertex not in vset:
                stack.append(ine.source_vertex)

    stack = [v]
    while stack:
        curv = stack.pop()
        vset.add(curv)
        for oute in curv.out_edges():
            if oute.target_vertex not in vset:
                stack.append(oute.target_vertex)

    return sorted(vset, key=line_pdg_sort_key)


def line_pdg_sort_key(x):
    f, ln, _ = x["name"].split(":")
    return f, int(ln)


def joern_parse(src):
    dst = src.rstrip("/") + "_parsed"
    shutil.rmtree(dst, ignore_errors=True)

    sp.call(
        " ".join(["./joern-parse", "-outdir", dst, "-outformat", "csv", src]),
        shell=True,
        cwd="./octopus-joern/",
    )

    sp.call("../octopus-joern-merge.sh", shell=True, cwd=dst)

    node_file = dst + "/nodes.csv"
    edge_file = dst + "/edges.csv"

    return node_file, edge_file


def get_slice_poi(cpg):
    """slice point of interest"""
    calls = set[int]()
    array_indexings = set[int]()
    ptr_member_accesses = set[int]()
    arithmatics = set[int]()

    for v in cpg.vs:
        ntype = v["type"].strip()
        if ntype == "CallExpression":
            calleev = v.graph.vs[v.index + 1]
            call_fn = calleev["code"]
            if call_fn in SENSI_API_SET:
                calls.add(v.index)
        elif ntype == "ArrayIndexing":
            array_indexings.add(v.index)
        elif ntype == "PtrMemberAccess":
            ptr_member_accesses.add(v.index)
        elif v["operator"] in {"+", "-", "*", "/"}:
            arithmatics.add(v.index)

    return calls | array_indexings | ptr_member_accesses | arithmatics


def get_slice_code(cpg, linepdg_vs):
    slice_code = {}
    for v in linepdg_vs:
        loc = v["name"]
        file, ln, _ = loc.split(":")
        ln = int(ln)
        vcode = v["code"]
        vcpgidx = v["cpgidx"]
        if vcode == "do {":
            # do-while loop, add the while condition to the slice
            dov = cpg.vs[vcpgidx]
            whilev = find_children_in_vtype(dov, "Condition")
            assert whilev
            whilevln = int(whilev["location"].split(":", 1)[0])
            slice_code[(file, whilevln)] = "} while ( " + whilev["code"] + " ) ;"
        slice_code[(file, ln)] = vcode

    slice_code = [code for _, code in sorted(slice_code.items())]
    return slice_code


def clean_slice(slice_code):
    user_funcs = {}
    var_symbols = {}

    fun_count = 1
    var_count = 1

    cleaned_slice = []
    for line in slice_code:
        # remove all string literals (keep the quotes)
        nostrlit_line = re.sub(r'".*?"', '""', line)
        # remove all character literals
        nocharlit_line = re.sub(r"'.*?'", "''", nostrlit_line)
        # replace any non-ASCII characters with empty string
        ascii_line = re.sub(r"[^\x00-\x7f]", r"", nocharlit_line)

        user_fun = REGX_FUNCALL.findall(ascii_line)
        user_var = REGX_VAR.findall(ascii_line)

        # Could easily make a "clean gadget" type class to prevent duplicate functionality
        # of creating/comparing symbol names for functions and variables in much the same way.
        # The comparison frozenset, symbol dictionaries, and counters would be class scope.
        # So would only need to pass a string list and a string literal for symbol names to
        # another function.

        for fun_name in user_fun:
            if (
                len({fun_name}.difference(MAIN)) != 0
                and len({fun_name}.difference(ALLKEYWORDS)) != 0
            ):
                # DEBUG
                # print('comparing ' + str(fun_name + ' to ' + str(main_set)))
                # print(fun_name + ' diff len from main is ' + str(len({fun_name}.difference(main_set))))
                # print('comparing ' + str(fun_name + ' to ' + str(keywords)))
                # print(fun_name + ' diff len from keywords is ' + str(len({fun_name}.difference(keywords))))
                ###
                # check to see if function name already in dictionary
                if fun_name not in user_funcs.keys():
                    user_funcs[fun_name] = "FUN" + str(fun_count)
                    fun_count += 1
                # ensure that only function name gets replaced (no variable name with same
                # identifier); uses positive lookforward
                ascii_line = re.sub(
                    r"\b(" + fun_name + r")\b(?=\s*\()",
                    user_funcs[fun_name],
                    ascii_line,
                )

        for var_name in user_var:
            # next line is the nuanced difference between fun_name and var_name
            if (
                len({var_name}.difference(ALLKEYWORDS)) != 0
                and len({var_name}.difference(MAINARGS)) != 0
            ):
                # DEBUG
                # print('comparing ' + str(var_name + ' to ' + str(keywords)))
                # print(var_name + ' diff len from keywords is ' + str(len({var_name}.difference(keywords))))
                # print('comparing ' + str(var_name + ' to ' + str(main_args)))
                # print(var_name + ' diff len from main args is ' + str(len({var_name}.difference(main_args))))
                ###
                # check to see if variable name already in dictionary
                if var_name not in var_symbols.keys():
                    var_symbols[var_name] = "VAR" + str(var_count)
                    var_count += 1
                # ensure that only variable name gets replaced (no function name with same
                # identifier); uses negative lookforward
                ascii_line = re.sub(
                    r"\b(" + var_name + r")\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()",
                    var_symbols[var_name],
                    ascii_line,
                )

        cleaned_slice.append(ascii_line)

    return cleaned_slice


if __name__ == "__main__":
    src = "/home/frezcirno/src/joern_slicer/test_project/"
    cpg_dst = src.rstrip("/") + "_cpg.picklez"
    line_pdg_dst = src.rstrip("/") + "_line_pdg.picklez"

    node_file, edge_file = joern_parse(src)

    cpg = load_graph(node_file, edge_file)
    cpg.write_picklez(cpg_dst)

    funcs = get_functions(cpg)
    add_call_edges(cpg, funcs)
    cpg.write_picklez(cpg_dst)

    line_pdg = merge_nodes(cpg)
    line_pdg.write_picklez(line_pdg_dst)

    with open("preview.txt", "w") as f:
        for v in cpg.vs:
            loc, info = get_top_node_info(v)
            if loc is None or info is None:
                print(" ==== ", v["type"], v["code"], file=f)
                continue
            print(loc, info["code"], " ==== ", v["type"], v["code"], file=f)

    #  Plot graph
    plot_dot(cpg, src.rstrip("/") + "_cpg.dot", render=True)
    plot_dot(line_pdg, src.rstrip("/") + "_line_pdg.dot", render=True)

    slice_points = get_slice_poi(cpg)

    slice_dataset = []
    for slice_point in slice_points:
        slice_pointv = cpg.vs[slice_point]

        loc, info = get_top_node_info(slice_pointv)
        linepdg_pointv = line_pdg.vs.find(name=loc)

        linepdg_vs = slice_graph_by(linepdg_pointv)
        slice_code = get_slice_code(cpg, linepdg_vs)

        slice_dataset.append((slice_code, slice_pointv["code"]))

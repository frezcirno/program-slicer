# %%
import pydot
from collections import defaultdict
import os
from igraph import Graph, Vertex
from csv import DictReader
from tqdm import tqdm
import Levenshtein

DIR = os.path.dirname(os.path.abspath(__file__))
SENSI_API_LIST = os.path.join(DIR, "resources", "sensiAPI.txt")
with open(SENSI_API_LIST, "r", encoding="utf-8") as f:
    SENSI_API_SET = frozenset([api.strip() for api in f.read().split(",")])

# CPPKEYWORDS up to C11 and C++17; immutable set
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


# %%


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


def find_parent_of_etype(v: Vertex, etype):
    if isinstance(etype, str):
        etype = [etype]
    for edge in v.in_edges():
        if edge["type"] in etype:
            return edge.source_vertex


def find_ast_parent(v: Vertex):
    return find_parent_of_etype(v, {"IS_AST_PARENT", "IS_FUNCTION_OF_AST", "IS_FILE_OF"})


def find_statement(v: Vertex):
    parent = v
    while parent:
        if parent["location"]:
            return parent
        parent = find_ast_parent(parent)


def find_function(v: Vertex):
    parent = v
    while parent:
        if parent["type"] == "Function":
            return parent
        parent = find_ast_parent(parent)


def find_file(v: Vertex):
    parent = v
    while parent:
        if parent["type"] == "File":
            return parent
        parent = find_ast_parent(parent)


def get_line_number(v: Vertex):
    parent = find_statement(v)
    if parent:
        return int(parent["location"].split(":", 1)[0])


def get_file(v: Vertex):
    parent = find_file(v)
    if parent:
        return parent["code"]


def get_location(v: Vertex):
    file = get_file(v)
    if file is None:
        return None
    line = get_line_number(v)
    if line is None:
        return None
    return f"{file}:{line}"


def get_callee(v: Vertex):
    vindex = v.index
    return v.graph.vs[vindex + 1]


def get_functions(cpg: Graph):
    funcs = {}
    for v in cpg.vs:
        if v["type"] == "CFGEntryNode":
            fnv = find_parent_of_etype(v, "IS_FUNCTION_OF_CFG")
            if fnv is None:
                continue
            filev = find_file(fnv)
            if filev is None:
                continue
            fname = fnv["code"]
            filename = filev["code"]
            funcs[(filename, fname)] = fnv.index
    return funcs


def add_inter_procedural_edges(cpg: Graph):
    funcs = get_functions(cpg)

    for v in cpg.vs:
        if v["type"] == "CallExpression":
            calleev = get_callee(v)
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
                matched_fns = sorted(matched_fns, key=lambda x: Levenshtein.distance(x[0], filename))
            matched_fn = matched_fns[0]

            cpg.add_edge(v, funcs[matched_fn], type="CALLS")


def get_location_code(location, basedir):
    file, ln = location.split(":")
    lastdir = os.path.basename(basedir.rstrip("/"))
    file = file.split(lastdir+"/", 1)[1]
    file = os.path.join(basedir, file)
    with open(file) as f:
        lines = f.readlines()
        return lines[int(ln) - 1].rstrip()


def get_line_graph(cpg: Graph, basedir):
    edges = []
    for e in cpg.es:
        if e["type"] in {"CONTROLS", "CALLS"}:
            src_key = get_location(e.source_vertex)
            dst_key = get_location(e.target_vertex)
            if not src_key or not dst_key or src_key == dst_key:
                continue
            edges.append((src_key, dst_key, "CDG", e["var"]))
        elif e["type"] == "REACHES":
            src_key = get_location(e.source_vertex)
            dst_key = get_location(e.target_vertex)
            if not src_key or not dst_key or src_key == dst_key:
                continue
            edges.append((src_key, dst_key, "DDG", e["var"]))

    edges = list(set(edges))

    pdg = Graph(directed=True)
    vs = []
    vsset = set()
    vattrs = {"code": []}
    es = []
    eattrs = {"type": [], "var": []}
    for src, dst, etype, var in edges:
        if src not in vsset:
            vs.append(src)
            vsset.add(src)
            vattrs["code"].append(get_location_code(src, basedir))
        if dst not in vsset:
            vs.append(dst)
            vsset.add(dst)
            vattrs["code"].append(get_location_code(dst, basedir))
        es.append((src, dst))
        eattrs["type"].append(etype)
        eattrs["var"].append(var)

    pdg.add_vertices(vs, vattrs)
    pdg.add_edges(es, eattrs)

    return pdg


def slice_cpg_by(pdgv: Vertex) -> list[Vertex]:
    pdgvset = set()

    stack = [pdgv]
    while stack:
        curpdgv = stack.pop()
        pdgvset.add(curpdgv)
        for ine in curpdgv.in_edges():
            if ine.source_vertex not in pdgvset:
                stack.append(ine.source_vertex)

    stack = [pdgv]
    while stack:
        curpdgv = stack.pop()
        pdgvset.add(curpdgv)
        for oute in curpdgv.out_edges():
            if oute.target_vertex not in pdgvset:
                stack.append(oute.target_vertex)

    return sorted(pdgvset, key=line_pdg_sort_key)


def line_pdg_sort_key(x):
    f, ln = x["name"].split(":")
    return f, int(ln)


# %%
node_file = "/home/frezcirno/src/joern_slicer/redis_parsed/nodes.csv"
edge_file = "/home/frezcirno/src/joern_slicer/redis_parsed/edges.csv"

cpg = load_graph(node_file, edge_file)

add_inter_procedural_edges(cpg)

cpg.write_picklez("cpg.picklez")

# %%
line_pdg = get_line_graph(cpg, "/home/frezcirno/src/joern_slicer/redis/")
line_pdg.write_picklez("line_pdg.picklez")

# %% Load from file
cpg = Graph.Read_Picklez("cpg.picklez")
line_pdg = Graph.Read_Picklez("line_pdg.picklez")

# %%
# plot_dot(cpg, render=True)
# plot_dot(line_pdg, "line_pdg.dot", render=True)

# %%
# slice point of interest
calls = set()
array_indexings = set()
ptr_member_accesses = set()
arithmatics = set()

for v in cpg.vs:
    ntype = v["type"].strip()
    if ntype == "CallExpression":
        calleev = get_callee(v)
        call_fn = calleev["code"]
        if call_fn in SENSI_API_SET:
            calls.add(v)
    elif ntype == "ArrayIndexing":
        array_indexings.add(v)
    elif ntype == "PtrMemberAccess":
        ptr_member_accesses.add(v)
    elif v["operator"] in {"+", "-", "*", "/"}:
        arithmatics.add(v)

slice_points = calls | array_indexings | ptr_member_accesses | arithmatics
slice_points_it = iter(slice_points)
dataset = []

# %%
cpg_slice_point = next(slice_points_it)
linepdg_slice_point = line_pdg.vs.find(name=get_location(cpg_slice_point))
slice_vs = slice_cpg_by(linepdg_slice_point)
print()
print("Slice at:", cpg_slice_point["code"])
for v in slice_vs:
    loc = v["name"]
    fn, ln = loc.split(":")
    print(ln, v["code"])

# %%
slice_subgraph = line_pdg.subgraph(slice_vs)
plot_dot(slice_subgraph, "slice_subgraph.dot", render=True)


# %%

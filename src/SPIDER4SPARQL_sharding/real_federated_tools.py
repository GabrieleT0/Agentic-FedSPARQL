import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

from rdflib import BNode, Graph, Literal, URIRef


_PREFIX_DECL_RE = re.compile(r"\bPREFIX\s+([^\s:]*)\s*:\s*<([^>]+)>", flags=re.IGNORECASE)
_PNAME_RE = re.compile(r"(?<![\w<])([A-Za-z_][\w\-]*)?:([A-Za-z_][\w\-]*)")


def _term_to_text(term):
    if term is None:
        return ""
    if hasattr(term, "n3"):
        return term.n3()
    return str(term)


def _extract_declared_prefixes(query: str):
    declared = set()
    for m in _PREFIX_DECL_RE.finditer(query):
        declared.add(m.group(1))
    return declared


def _extract_used_prefixes(query: str):
    used = set()
    for m in _PNAME_RE.finditer(query):
        used.add(m.group(1) or "")
    return used


def _infer_default_namespace(graph: Graph):
    for prefix, ns in graph.namespaces():
        if prefix in (None, ""):
            return str(ns)

    for s, _p, _o in graph:
        if isinstance(s, URIRef):
            uri = str(s)
            if uri.startswith("http://www.w3.org/"):
                continue
            if "#" in uri:
                return uri.rsplit("#", 1)[0] + "#"
            if "/" in uri:
                return uri.rsplit("/", 1)[0] + "/"
    return None


def _prepare_query_with_prefixes(query: str, graph: Graph):
    declared = _extract_declared_prefixes(query)
    used = _extract_used_prefixes(query)

    graph_prefix_map = {}
    for prefix, ns in graph.namespaces():
        graph_prefix_map[(prefix or "")] = str(ns)

    missing = []
    for pref in sorted(used):
        if pref in declared:
            continue
        if pref == "":
            ns = graph_prefix_map.get("") or _infer_default_namespace(graph)
            if ns:
                missing.append(f"PREFIX : <{ns}>")
        else:
            ns = graph_prefix_map.get(pref)
            if ns:
                missing.append(f"PREFIX {pref}: <{ns}>")

    if not missing:
        return query
    return "\n".join(missing) + "\n" + query


def _extract_prefix_lines(query: str):
    lines = []
    for line in query.splitlines():
        if line.strip().upper().startswith("PREFIX "):
            lines.append(line.strip())
    return lines


def _term_to_sparql_json(term):
    if isinstance(term, URIRef):
        return {"type": "uri", "value": str(term)}
    if isinstance(term, BNode):
        return {"type": "bnode", "value": str(term)}
    if isinstance(term, Literal):
        binding = {"type": "literal", "value": str(term)}
        if term.language:
            binding["xml:lang"] = term.language
        if term.datatype:
            binding["datatype"] = str(term.datatype)
        return binding
    return {"type": "literal", "value": str(term)}


def _parse_select_vars(query: str):
    m = re.search(r"\bSELECT\b\s+(DISTINCT\s+)?(.+?)\s+\bWHERE\b", query, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    raw = m.group(2).strip()
    if raw == "*":
        return ["*"]
    return re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", raw)


def _normalize_select_result(query_result):
    vars_ = [f"?{str(v)}" for v in getattr(query_result, "vars", [])]
    rows = []
    for binding in query_result.bindings:
        row = {}
        for var in query_result.vars:
            var_key = f"?{str(var)}"
            val = binding.get(var)
            if val is not None:
                row[var_key] = _term_to_text(val)
        rows.append(row)
    return {"type": "SELECT", "vars": vars_, "rows": rows}


def execute_query_local(graph: Graph, sparql: str):
    prepared_query = _prepare_query_with_prefixes(sparql, graph)
    result = graph.query(prepared_query)
    result_type = getattr(result, "type", "SELECT")

    if result_type == "ASK":
        return {"type": "ASK", "value": bool(result.askAnswer)}
    if result_type == "SELECT":
        return _normalize_select_result(result)

    g = Graph()
    for triple in result.graph:
        g.add(triple)
    serialized = g.serialize(format="nt")
    return {"type": str(result_type), "graph_nt": serialized}


class _ShardSPARQLEndpointHandler(BaseHTTPRequestHandler):
    graphs_by_dataset = {}

    def log_message(self, format, *args):
        return

    def _write_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/sparql-results+json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]

        if len(parts) < 2 or parts[-1] != "sparql":
            self._write_json({"error": "Invalid path. Use /<dataset>/sparql"}, status=404)
            return

        dataset = "/".join(parts[:-1])
        graph = self.graphs_by_dataset.get(dataset)
        if graph is None:
            self._write_json({"error": f"Unknown dataset: {dataset}"}, status=404)
            return

        query = parse_qs(parsed.query).get("query", [None])[0]
        if not query:
            self._write_json({"error": "Missing 'query' parameter"}, status=400)
            return

        try:
            prepared_query = _prepare_query_with_prefixes(query, graph)
            result = graph.query(prepared_query)
            result_type = getattr(result, "type", "SELECT")

            if result_type == "ASK":
                self._write_json({"head": {}, "boolean": bool(result.askAnswer)})
                return

            vars_ = [str(v) for v in result.vars]
            bindings = []
            for row in result.bindings:
                row_payload = {}
                for var in result.vars:
                    value = row.get(var)
                    if value is not None:
                        row_payload[str(var)] = _term_to_sparql_json(value)
                bindings.append(row_payload)

            self._write_json({"head": {"vars": vars_}, "results": {"bindings": bindings}})
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=400)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class LocalFederationServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 18080, request_host: str | None = None):
        self.host = host
        self.request_host = request_host or ("127.0.0.1" if host == "0.0.0.0" else host)
        self.server = _ReusableThreadingHTTPServer((host, port), _ShardSPARQLEndpointHandler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def set_shards(self, shards_info: dict, dataset_prefix: str = "") -> dict:
        prefix = dataset_prefix.strip("/")
        graphs = {}
        deployed = {}

        for cls, info in shards_info.items():
            dataset = info.get("class_name", str(cls))
            dataset_key = f"{prefix}/{dataset}" if prefix else dataset
            path = info.get("path")
            if not path:
                continue

            g = Graph()
            g.parse(path)
            graphs[dataset_key] = g

            cloned = dict(info)
            cloned["endpoint_url"] = f"http://{self.request_host}:{self.port}/{dataset_key}/sparql"
            deployed[cls] = cloned

        _ShardSPARQLEndpointHandler.graphs_by_dataset = graphs
        return deployed

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def start_local_federation_endpoints(
    shards_info: dict,
    host: str = "127.0.0.1",
    port: int = 18080,
    dataset_prefix: str = "",
):
    federation_server = LocalFederationServer(host=host, port=port)
    deployed = federation_server.set_shards(shards_info, dataset_prefix=dataset_prefix)
    return federation_server, deployed


def _extract_service_blocks(query: str):
    blocks = []
    pos = 0

    while True:
        m = re.search(r"\bSERVICE\b\s*<([^>]+)>\s*\{", query[pos:], flags=re.IGNORECASE)
        if not m:
            break

        start = pos + m.start()
        endpoint = m.group(1)
        open_brace = query.find("{", start)

        i = open_brace
        depth = 0
        close_brace = -1
        while i < len(query):
            ch = query[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_brace = i
                    break
            i += 1

        if close_brace == -1:
            break

        inner = query[open_brace + 1 : close_brace].strip()
        blocks.append({"endpoint": endpoint, "body": inner})
        pos = close_brace + 1

    return blocks


def _parse_sparql_json_rows(payload: dict):
    vars_ = [f"?{v}" for v in payload.get("head", {}).get("vars", [])]
    rows = []
    for row in payload.get("results", {}).get("bindings", []):
        normalized = {}
        for var, binding in row.items():
            normalized[f"?{var}"] = binding.get("value", "")
        rows.append(normalized)
    return vars_, rows


def _natural_join(left_rows, right_rows):
    if not left_rows:
        return list(right_rows)
    if not right_rows:
        return []

    joined = []
    for l in left_rows:
        for r in right_rows:
            shared = set(l.keys()) & set(r.keys())
            if all(l[k] == r[k] for k in shared):
                merged = dict(l)
                merged.update(r)
                joined.append(merged)
    return joined


def execute_federated_query_real(federated_sparql: str, timeout: float = 20.0):
    blocks = _extract_service_blocks(federated_sparql)
    if not blocks:
        raise ValueError("No SERVICE blocks found in federated query.")

    select_vars = _parse_select_vars(federated_sparql)
    prefix_lines = _extract_prefix_lines(federated_sparql)
    prefix_block = ("\n".join(prefix_lines) + "\n") if prefix_lines else ""
    partial_rows = [{}]

    for block in blocks:
        subquery = prefix_block + f"SELECT * WHERE {{\n{block['body']}\n}}"
        query_string = urlencode({"query": subquery})
        url = f"{block['endpoint']}?{query_string}"
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        _vars, rows = _parse_sparql_json_rows(payload)
        partial_rows = _natural_join(partial_rows, rows)

    if select_vars == ["*"] or not select_vars:
        vars_ = sorted({k for row in partial_rows for k in row.keys()})
    else:
        vars_ = select_vars

    projected = []
    for row in partial_rows:
        projected.append(tuple(row.get(v, "") for v in vars_))
    projected.sort()

    return {"type": "SELECT", "vars": vars_, "rows": projected}


def validate_rewrite_real_federated(
    original_sparql: str,
    federated_sparql: str,
    full_kg_path: str | None = None,
    full_graph: Graph | None = None,
) -> dict:
    if full_graph is None:
        if not full_kg_path:
            raise ValueError("Either full_kg_path or full_graph must be provided.")
        full_graph = Graph()
        full_graph.parse(full_kg_path)

    original = execute_query_local(full_graph, original_sparql)
    federated = execute_federated_query_real(federated_sparql)

    original_vars = original.get("vars", [])
    federated_vars = federated.get("vars", [])

    original_rows = []
    for row in original.get("rows", []):
        original_rows.append(tuple(row.get(v, "") for v in original_vars))
    original_rows.sort()

    equivalent = (
        original.get("type") == federated.get("type")
        and original_vars == federated_vars
        and original_rows == federated.get("rows")
    )

    return {
        "equivalent": equivalent,
        "original_result": original,
        "federated_result": federated,
    }

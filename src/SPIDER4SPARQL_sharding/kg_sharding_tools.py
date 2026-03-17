from collections import defaultdict
from pathlib import Path
import re

from rdflib import Graph, RDF
from rdflib import Graph as RDFGraph


def _sanitize_turtle_text(ttl_text: str) -> str:
    # Some SPIDER4SPARQL files contain URI refs like <...\#...> which are invalid.
    # Normalizing \# to # is enough to recover most affected files.
    return re.sub(r"<([^>]*)>", lambda m: "<" + m.group(1).replace("\\#", "#") + ">", ttl_text)


def _load_graph_resilient(kg_path: str) -> Graph:
    g = Graph()
    try:
        g.parse(kg_path, format="turtle")
        return g
    except Exception:
        ttl_text = Path(kg_path).read_text(encoding="utf-8", errors="replace")
        sanitized = _sanitize_turtle_text(ttl_text)
        g = Graph()
        g.parse(data=sanitized, format="turtle")
        return g


def analyze_kg(kg_path: str) -> dict:
    """Extract classes and their triples from a KG.

    Returns the analysis dict plus a ``graph`` key holding the loaded rdflib
    Graph so callers can reuse it without re-parsing the file.
    """
    g = _load_graph_resilient(kg_path)

    subject_to_class = {}
    for s, _p, o in g.triples((None, RDF.type, None)):
        if str(o).startswith("http://www.w3.org/"):
            continue
        subject_to_class[s] = o

    class_triples = defaultdict(list)
    untyped_triples = []

    for s, p, o in g:
        if s in subject_to_class:
            cls = subject_to_class[s]
            class_triples[cls].append((s, p, o))
        else:
            untyped_triples.append((s, p, o))

    return {
        "classes": dict(class_triples),
        "untyped": untyped_triples,
        "subject_to_class": subject_to_class,
        "total_triples": len(g),
        "graph": g,
    }


def shard_by_class(kg_path: str, output_dir: str, analysis: dict | None = None) -> dict:
    """Split a KG into class-based shards and return shard metadata.

    ``analysis`` can be passed in from a prior ``analyze_kg`` call to avoid
    loading the KG a second time.  When omitted the KG is loaded from
    ``kg_path`` as before.
    """
    if analysis is None:
        analysis = analyze_kg(kg_path)
    shards = {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reuse the already-loaded graph from analysis to avoid a second parse.
    original = analysis.get("graph") or _load_graph_resilient(kg_path)
    namespace_bindings = list(original.namespaces())

    for cls_uri, triples in analysis["classes"].items():
        shard = RDFGraph()
        for prefix, ns in namespace_bindings:
            shard.bind(prefix, ns)

        for s, p, o in triples:
            shard.add((s, p, o))

        for _s, _p, o in triples:
            if o in analysis["subject_to_class"]:
                obj_class = analysis["subject_to_class"][o]
                if obj_class == cls_uri:
                    shard.add((o, RDF.type, obj_class))

        class_name = str(cls_uri).split("/")[-1].split("#")[-1]
        shard_path = output_path / f"{class_name}.ttl"
        shard.serialize(str(shard_path), format="turtle")

        shards[cls_uri] = {
            "path": str(shard_path),
            "class_name": class_name,
            "triple_count": len(shard),
            "endpoint_url": f"http://localhost:3030/{class_name}/sparql",
        }

    if analysis["untyped"]:
        shared = RDFGraph()
        for prefix, ns in namespace_bindings:
            shared.bind(prefix, ns)
        for s, p, o in analysis["untyped"]:
            shared.add((s, p, o))

        shared_path = output_path / "_shared.ttl"
        shared.serialize(str(shared_path), format="turtle")
        shards["_shared"] = {
            "path": str(shared_path),
            "class_name": "_shared",
            "triple_count": len(shared),
            "endpoint_url": "http://localhost:3030/_shared/sparql",
        }

    return shards

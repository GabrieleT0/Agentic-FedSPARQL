# SPIDER4FedSPARQL Dataset

This folder contains SPIDER4FedSPARQL data generated from SPIDER4SPARQL for
federated SPARQL question answering. The data is organized into sharding
configurations that split source RDF knowledge graphs into smaller SPARQL
endpoint shards and, where available, pair natural-language questions with
rewritten federated SPARQL queries.

Use this README as the single dataset-level description for the
`SPIDER4FedSPARQL` folder.

## Folder Layout

| Path | Description |
| --- | --- |
| `class-sharding/` | Main benchmark-ready dataset. RDF graphs are split by RDF class, and benchmark examples include federated `SERVICE`-based SPARQL rewrites. |
| `class-predicate-sharding/` | Experimental class+predicate shard layout. It contains generated shards and endpoint mappings, but the benchmark JSON files are empty in this snapshot. |
| `shards/` | Top-level copy of the class-sharding RDF/Turtle shards, retained for compatibility with earlier packaging layouts. |

## Main Dataset: `class-sharding/`

The class-sharding configuration is the usable benchmark release in this
folder. Each source knowledge graph is partitioned into class-level Turtle
files, and each retained question contains both the original SPARQL query and a
federated rewrite over local shard endpoints.

### Summary

| Item | Count |
| --- | ---: |
| Benchmark examples | 2,229 |
| Training examples | 1,884 |
| Development examples | 345 |
| Knowledge graphs represented in benchmark examples | 131 |
| Class-shard endpoint records | 661 |
| Unique class URIs in endpoint records | 519 |
| RDF/Turtle shard files | 661 |

### Files

| Path | Description |
| --- | --- |
| `class-sharding/SPIDER4FedSPARQL_benchmark.json` | Full class-sharding benchmark, containing train and dev examples. |
| `class-sharding/SPIDER4FedSPARQL_train.json` | Training split. |
| `class-sharding/SPIDER4FedSPARQL_dev.json` | Development split. |
| `class-sharding/shards/` | Class-level RDF/Turtle shards, stored as `shards/<kg_name>/<class_name>.ttl`. |
| `class-sharding/shard_endpoints.json` | Mapping from each class shard to its intended local SPARQL endpoint. |
| `class-sharding/endpoints_metadata.json` | Endpoint metadata for discovery, schema retrieval, and query planning. |
| `class-sharding/progress.json` | Generation and validation summary statistics. |

### Benchmark JSON Schema

Each benchmark JSON file is a list of examples with the following fields:

```json
{
  "id": "activity_1_1",
  "kg_name": "activity_1",
  "split": "train",
  "question": "What are the first name and last name of Linda Smith's advisor?",
  "original_sparql": "SELECT ... WHERE { ... }",
  "federated_sparql": "PREFIX ... SELECT ... WHERE { SERVICE <...> { ... } }",
  "endpoints": [
    {
      "class": "http://valuenet/ontop/faculty",
      "url": "http://host.docker.internal:3030/activity_1__faculty/sparql"
    }
  ],
  "validation": {
    "valid": true,
    "original_count": 7,
    "federated_count": 7,
    "missing": 0,
    "extra": 0
  }
}
```

Field descriptions:

- `id`: Stable benchmark example identifier.
- `kg_name`: Source knowledge graph name.
- `split`: Dataset split, either `train` or `dev`.
- `question`: Natural-language question.
- `original_sparql`: Original non-federated SPIDER4SPARQL query.
- `federated_sparql`: Rewritten federated query using SPARQL `SERVICE`
  clauses over class-level endpoints.
- `endpoints`: Candidate local class-shard endpoints associated with the
  source knowledge graph.
- `validation`: Result of comparing the original query with the federated
  rewrite when local endpoints were deployed.

### Endpoint Metadata

`class-sharding/shard_endpoints.json` contains one record per class shard:

```json
{
  "kg_name": "activity_1",
  "class": "http://valuenet/ontop/participates_in",
  "class_name": "participates_in",
  "endpoint_name": "activity_1__participates_in",
  "endpoint_url": "http://host.docker.internal:3030/activity_1__participates_in/sparql"
}
```

`class-sharding/endpoints_metadata.json` contains retrieval-oriented metadata:

```json
{
  "url": "http://host.docker.internal:3030/activity_1__activity/sparql",
  "classes": ["http://valuenet/ontop/activity"],
  "properties": ["http://valuenet/ontop/activity#actid"],
  "examples": ["SELECT ?s ?o WHERE { ... } LIMIT 10"],
  "description": "This knowledge graph models activities as the primary domain entities..."
}
```

## Experimental Dataset: `class-predicate-sharding/`

The class+predicate-sharding folder contains a more granular shard layout in
which class shards are further divided by predicate group. In this snapshot,
the generated shard files and endpoint index are present, but the benchmark
split files are empty arrays.

| Item | Count |
| --- | ---: |
| Benchmark examples | 0 |
| Endpoint records | 4,197 |
| Knowledge graphs represented in endpoint records | 134 |
| Unique class URIs in endpoint records | 498 |
| Predicate groups | 1,857 |
| RDF/Turtle shard files | 4,197 |

Files in this configuration:

| Path | Description |
| --- | --- |
| `class-predicate-sharding/SPIDER4FedSPARQL_CP_benchmark.json` | Empty benchmark file in this snapshot. |
| `class-predicate-sharding/SPIDER4FedSPARQL_CP_train.json` | Empty training split in this snapshot. |
| `class-predicate-sharding/SPIDER4FedSPARQL_CP_dev.json` | Empty development split in this snapshot. |
| `class-predicate-sharding/shards/` | Class+predicate RDF/Turtle shards. |
| `class-predicate-sharding/shard_endpoints.json` | Mapping from class+predicate shards to intended local SPARQL endpoints. |
| `class-predicate-sharding/progress.json` | Generation summary. |

Example endpoint record:

```json
{
  "kg_name": "activity_1",
  "shard_key": "http://valuenet/ontop/participates_in::ref_actid",
  "class_uri": "http://valuenet/ontop/participates_in",
  "class_name": "participates_in",
  "predicate_group": "ref_actid",
  "endpoint_name": "activity_1__participates_in__ref_actid",
  "endpoint_url": "http://host.docker.internal:3030/activity_1__participates_in__ref_actid/sparql"
}
```

## Validation Status for `class-sharding/`

Validation compares the result set of `original_sparql` on the full source
graph with the result set of `federated_sparql` over the deployed class shards.

| Status | Examples |
| --- | ---: |
| `valid: true` | 1,677 |
| `error: "result_mismatch"` | 340 |
| `error: "execution_failed"` | 187 |
| `error: "endpoints_not_deployed"` | 25 |

For strict answer-equivalence evaluation, filter to:

```python
example["validation"].get("valid") is True
```

For endpoint discovery, schema selection, query planning, and robustness
experiments, the full class-sharding benchmark may be useful because it retains
failed and mismatched cases with explicit diagnostics.

## Running Federated Queries Locally

Endpoint URLs are local Apache Jena Fuseki URLs of the form:

```text
http://host.docker.internal:3030/<endpoint_name>/sparql
```

These are not public hosted endpoints. To execute the provided federated
queries:

1. Start a local SPARQL server, such as Apache Jena Fuseki.
2. Load each `.ttl` shard as a separate dataset.
3. Use the corresponding `endpoint_name` from `shard_endpoints.json` as the
   dataset name.
4. Keep the `host.docker.internal:3030` address, or rewrite the endpoint URLs
   in the JSON files to match your deployment.
5. Execute `federated_sparql` through an endpoint that supports SPARQL
   `SERVICE` federation.

The generation pipeline used one Fuseki dataset per shard and an empty dataset
named `SPIDER4FedSPARQL` as the federated query executor. Because rewritten
queries place graph patterns inside `SERVICE` clauses, the executor dataset
itself does not need to contain RDF triples.

## Intended Use

This dataset is intended for research on:

- natural-language question answering over federated RDF graphs;
- SPARQL endpoint discovery and source selection;
- schema-aware federated query planning;
- federated SPARQL generation from natural language;
- validation and robustness analysis for query rewriting.

## Provenance

SPIDER4FedSPARQL is generated from SPIDER4SPARQL. The class-sharding benchmark
retains the original natural-language questions and original SPARQL queries,
then adds federated SPARQL rewrites over class-level shard endpoints.

## Citation

Please cite the Zenodo record associated with this dataset. If your work relies
on the original benchmark content, also cite the original SPIDER4SPARQL source
as appropriate.

## License

Use of this dataset is governed by the license stated on the Zenodo record. The
associated software repository includes an MIT License for code components.

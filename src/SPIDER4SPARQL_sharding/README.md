# SPIDER4FedSPARQL

A utility to build federated SPARQL benchmarks from class-sharded knowledge graphs.

Real federated mode (without Fuseki): shards are exposed as local HTTP SPARQL endpoints, and federated queries with `SERVICE` are executed against those endpoints.

## Run with a Single Entry Point

The main entry point is located at `src/SPIDER4SPARQL_sharding/main.py`.

The pipeline automatically uses the dataset in `data/original_SPIDER4SPARQL` (with fallback to `data/original_SPIEDER4SPARQL`).

Run the full pipeline with a single command:

```bash
python src/SPIDER4SPARQL_sharding/main.py
```

The main script performs the following steps:
- load SPIDER4SPARQL examples from `dev/dev.json` and `train/train.json`
- shard each KG domain by class
- start local SPARQL endpoints for each shard
- rewrite queries into federated form using `SERVICE`
- validate result equivalence (original query on full KG vs real federated query on local endpoints)
- save federated queries and shard metadata
- package the final benchmark as JSON

## Quick Smoke Test

To validate the end-to-end pipeline on a small subset:

```bash
python src/SPIDER4SPARQL_sharding/main.py --smoke-test
```

By default, `--smoke-test` runs only on `dev` with 3 examples.
You can override this limit:

```bash
python src/SPIDER4SPARQL_sharding/main.py --smoke-test --max-examples-per-split 20
```

## Main Output

- `data/federated_output/benchmark_spider4sparql_real_federated.json`
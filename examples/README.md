# Examples

Reusable configuration artifacts referenced by the framework.

## `hermes_tools_evolution_metadata.json`

A `_evolution_metadata.json` sidecar declaring confusable-neighbor pairs for
the user's `hermes-agent/tools/` directory. The dataset builder reads
`<tools_dir>/_evolution_metadata.json` to learn which tools are nearest
behavioral neighbors of which; without it, the confusable-neighbor bucket
(`--enable-confusable-bucket`) silently reallocates its examples to
`target_correct` and the eval never probes the boundary GEPA needs to
disambiguate.

Copy this file into your hermes-agent checkout:

```
cp examples/hermes_tools_evolution_metadata.json \
   <hermes-agent>/tools/_evolution_metadata.json
```

The declarations are bidirectional — `"write_file": "patch"` and
`"patch": "write_file"` both need entries.

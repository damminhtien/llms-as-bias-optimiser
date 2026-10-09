# V3 Pilot Report

## Gate

- Decision: **GO**.
- Candidates: 40/40.
- Complete generations: {'0': 10, '1': 15, '2': 15}.
- LLM proposals outside the global-histogram family: 29/30 (96.7%).
- Predeclared rule: 40 complete records and at least 30% of valid LLM proposals, excluding hand-built seeds, outside the global-histogram family.
- Evaluated train sizes: [500, 5000].
- LLM model: qwen3.5:35b-mlx.
- Archive SHA-256: `1aa98f8106b8bc6a553ebcf80e84c2b9afacc5b4117086a4a634b870cd201672`.
- Response archive SHA-256: `2d470c3a694b6a1913296995ad1cf29fecd985b81c4fd37b09f09f5566fdc830`.
- Failure archive SHA-256: `3d871dd46dfb96f3d4474f71b74bc433448efb25660330b44e5997f742fa9063`.

## Search health

- Occupied descriptor cells: 17.
- Invalid AST rate: 11.1% (5/45 attempts).
- Duplicate proposal rate: 13.3% (6/45 attempts).
- Feature width min / median / max: 1 / 24.0 / 80.
- Order-sensitive candidates: 8.
- Relational candidates: 19.
- Rejected for mechanism mismatch: 4.
- Rejected for evaluation errors: 0.
- Accepted mechanism families: {'free_exploration': 6, 'graph_relational': 8, 'order_sensitive': 8, 'spatial_relational': 8}.
- Requested proposal slots, including refills: {'free_exploration': 20, 'graph_relational': 8, 'order_sensitive': 9, 'spatial_relational': 8}.

## Descriptor cells

- `('graph', 'first_order', 'global', 'single', 'moderate')`
- `('graph', 'first_order', 'localized', 'single', 'moderate')`
- `('graph', 'higher_order', 'global', 'single', 'moderate')`
- `('graph', 'orderless', 'global', 'composite', 'deep')`
- `('graph', 'orderless', 'global', 'single', 'moderate')`
- `('mixed', 'orderless', 'global', 'composite', 'deep')`
- `('mixed', 'orderless', 'global', 'composite', 'moderate')`
- `('mixed', 'orderless', 'localized', 'composite', 'deep')`
- `('path_geometry', 'first_order', 'global', 'single', 'moderate')`
- `('path_geometry', 'first_order', 'localized', 'composite', 'deep')`
- `('path_geometry', 'first_order', 'localized', 'single', 'moderate')`
- `('path_geometry', 'higher_order', 'global', 'composite', 'deep')`
- `('path_geometry', 'higher_order', 'global', 'single', 'moderate')`
- `('path_geometry', 'orderless', 'global', 'single', 'moderate')`
- `('path_geometry', 'orderless', 'localized', 'composite', 'deep')`
- `('path_geometry', 'orderless', 'localized', 'single', 'moderate')`
- `('topology', 'orderless', 'global', 'single', 'compact')`

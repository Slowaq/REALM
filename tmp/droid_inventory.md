# DROID inventory (local machine, 2026-10-07)

Search: `/home/elmo/sedlam/REALM/data/droid` first, then `find / -maxdepth 7 -type d -name 'chunk-*' | grep -i droid` (plus a depth-8 check).

## Dataset roots

| root | chunk-* folders | episode_*.parquet | du -sh | meta/ |
|---|---|---|---|---|
| `/home/elmo/sedlam/REALM/data/droid_1.0.1` | 1 (`chunk-000`) | 1000 (`episode_000000`..`episode_000999`) | 1.8G total (chunk-000: 123M, videos/: 1.6G, extracted_eps/: 52M, chunk-000.zip: 80M) | **missing**. No info.json, episodes.jsonl or tasks.jsonl |

`/home/elmo/sedlam/REALM/data/droid/` has no chunks. It holds only `DROID100_tabletop.json` (15 KB).

## Duplicates and non-data folders (each chunk counted once)
- `droid_1.0.1/chunk-000.zip` (83 MB) is an archive of the same `chunk-000/` (1000 parquets, 126 MB uncompressed). It is a duplicate and was not counted.
- `droid_1.0.1/extracted_eps/chunk-000/episode_*/` contains derived `.npy` arrays (actions and states), no parquets. Not counted.
- `droid_1.0.1/videos/chunk-000/` contains only camera videos. Not counted.
- `data/datasets/omnigibson-robot-assets/models/droid*` are robot models, not episodes.

## Schema check (episode_000000.parquet)
1 row group, 167 rows. The columns `language_instruction`, `language_instruction_2`, `language_instruction_3` and `task_category` are strings, and `is_episode_successful` is a bool. All are present. `task_category` holds a location string, for example "2479 Richard Ct". Checked with the base conda env (pyarrow 23.0.0).

## Coverage
1 chunk, 1000 episodes, 838 successful, 0 unreadable. 207 episodes have no instruction at all. 2080 unique instructions. 50 distinct locations among successful episodes.

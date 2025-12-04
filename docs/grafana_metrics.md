# Grafana Telemetry for Handover DQN

This repo now writes two SQLite tables inside each simulation folder (`output/<run-id>/database.db`):

| Table | Purpose | Columns |
| --- | --- | --- |
| `ho_reward_metrics` | Reward vs. step | `timestamp`, `ueImsiComplete` (step id), `step`, `reward` |
| `ho_training_metrics` | DQN loss/epsilon vs. step | `timestamp`, `ueImsiComplete` (step id), `step`, `loss`, `epsilon` |

Both tables are created by `HandoverEnv._init_datalake_usecase`, and the environment/example automatically populate them each step.

---

## Step-by-step instructions

### 1. Run the simulation to generate the SQLite DB

```bash
cd /home/ubadmin/ns-o-ran-gym
python examples/handover.py \
  --config src/environments/scenario_configurations/ts_use_case.json \
  --output_folder output \
  --ns3_path /home/ubadmin/ns3-mmwave-oran
```

Each run creates a folder like `output/24fdaf51-6ccb-4a8e-af49-7c9192f712a8/` containing `database.db`.

### 2. Quick verification via sqlite3 (optional)

```bash
cd /home/ubadmin/ns-o-ran-gym/output/<run-id>
sqlite3 database.db
```

Inside the prompt:

```sql
.tables
SELECT * FROM ho_reward_metrics LIMIT 5;
SELECT step, loss, epsilon FROM ho_training_metrics ORDER BY step DESC LIMIT 5;
.quit
```

---

## Connect Grafana to the DB

### 3. Install Grafana plugin and restart Grafana

```bash
sudo grafana-cli plugins install frser-sqlite-datasource
sudo systemctl restart grafana-server
```

Open [http://localhost:3000](http://localhost:3000) in a browser and log in (default admin/admin).

### 4. Add SQLite data source

1. Grafana sidebar → **Configuration** → **Data sources** → **Add data source**.
2. Select **SQLite**.
3. Set *Path* to `/home/ubadmin/ns-o-ran-gym/output/<run-id>/database.db`.
4. Click **Save & Test**.

### 5. Create dashboards

1. Grafana sidebar → **Dashboards** → **New → Dashboard → Add visualization**.
2. Choose the SQLite source and enter a query, e.g.:

```sql
SELECT step, reward
FROM ho_reward_metrics
ORDER BY step;
```

```sql
SELECT step, loss, epsilon
FROM ho_training_metrics
ORDER BY step;
```

3. Set visualization to **Time series**, with `step` on the X-axis and the metric you’re plotting on Y.
4. In the dashboard toolbar, set the refresh interval (e.g. `5s`) so the panel re-queries while the run is in progress.

---

## Aggregate every run into one database (recommended)

Instead of re-pointing Grafana to each new UUID folder, run the aggregator while training. It copies the per-run metrics into a single SQLite file (`output/grafana_metrics.db`) that Grafana follows forever.

### 6. Start the aggregator alongside training

```bash
cd /home/ubadmin/ns-o-ran-gym
python scripts/metrics_aggregator.py \
  --output-root /home/ubadmin/ns-o-ran-gym/output \
  --central-db /home/ubadmin/ns-o-ran-gym/output/grafana_metrics.db \
  --poll-interval 2
```

Leave this terminal running; it scans every UUID folder, reads `ho_reward_metrics` and `ho_training_metrics`, and mirrors them into the central database with an extra `run_id` column.

### 7. Point Grafana at the aggregated DB

Reuse steps 3–5 above, but set the SQLite *Path* to `/home/ubadmin/ns-o-ran-gym/output/grafana_metrics.db`. All dashboards now stay connected even when new runs appear, and you can filter per `run_id` in your SQL queries, e.g.:

```sql
SELECT timestamp, reward
FROM dqn_reward_metrics
WHERE run_id = '${run_id}'
ORDER BY step;
```

To show every run on the same panel, drop the `WHERE` clause and use Grafana's legend/transform features.


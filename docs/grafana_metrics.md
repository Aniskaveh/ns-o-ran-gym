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

> Note: every `env.reset()` or new run creates a new output UUID. Update the data source path to point Grafana at the latest `/output/<run-id>/database.db`.


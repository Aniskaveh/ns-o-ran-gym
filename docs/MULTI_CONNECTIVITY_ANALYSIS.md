# Multi-Connectivity Analysis for Scenario-Ten

## Summary

**YES, UEs in scenario_ten have multi-connectivity:**
- **LTE Anchor (cellId = 1)**: All UEs maintain a **persistent connection** to the LTE eNB
- **gNB Secondary (cellIds = 2-8)**: UEs also connect to one gNB at a time, which can be handed over

**However, the current HO-GRL implementation only tracks and controls the gNB connection, not the LTE anchor.**

---

## Evidence from Data

### 1. LTE Connection (Always Maintained)

**From `lte_cu_cp` table (cu-cp-cell-1.txt):**
- **All 14 UEs** have entries in `lte_cu_cp` table
- Each entry shows:
  - `cellId = 1` (LTE anchor)
  - `sameCellSinr`: SINR to LTE anchor (e.g., 23-35 dB)
  - `numActiveUes = 14`: All UEs are connected to LTE

**Example:**
```
timestamp=1765812797761, ueImsiComplete=1, cellId=1, sameCellSinr=24.116722
timestamp=1765812797761, ueImsiComplete=2, cellId=1, sameCellSinr=27.833485
...
(all 14 UEs have entries)
```

**This confirms**: UEs maintain **persistent connection to LTE anchor**.

### 2. gNB Connection (Controlled by Agent)

**From `gnb_cu_cp` table (cu-cp-cell-2.txt to cu-cp-cell-8.txt):**
- **All 14 UEs** have entries in `gnb_cu_cp` table
- Each entry shows:
  - `L3 serving Id(m_cellId)`: The gNB cell ID (2-8) that UE is connected to
  - `L3 serving SINR`: SINR to the serving gNB
  - `L3 neigh SINR 1-6`: SINR to neighbor gNBs

**From `CellIdStats.txt`:**
- Shows only cells **2-8** (gNBs), never cell 1 (LTE)
- This tracks the **gNB serving cell**, not LTE

**Example:**
```
0.0542144 2 3 2  # UE 2 connected to gNB 3, 2 handovers so far
0.0542145 3 6 1  # UE 3 connected to gNB 6, 1 handover so far
```

**This confirms**: UEs connect to **one gNB at a time**, which can be handed over.

### 3. No UEs with LTE as Serving Cell in gNB Table

**Query result:**
```sql
SELECT COUNT(*) FROM gnb_cu_cp WHERE "L3 serving Id(m_cellId)" = 1;
-- Result: 0
```

**This confirms**: No UE has LTE (cellId=1) as their **serving cell** in the gNB context. The serving cell is always a gNB (2-8).

---

## How Multi-Connectivity Works in Scenario-Ten

### Architecture:
```
UE
├── LTE Anchor (cellId=1) ──► Always connected, not controlled by agent
│   └── Control plane: lte_cu_cp
│   └── User plane: lte_cu_up
│   └── Purpose: Anchor connection for reliability
│
└── gNB Secondary (cellId=2-8) ──► Controlled by agent (handover decisions)
    └── Control plane: gnb_cu_cp
    └── User plane: gnb_cu_up
    └── Purpose: High-throughput data connection
```

### Connection Types:
1. **LTE Anchor (cellId=1)**:
   - **Always maintained** (not controlled by agent)
   - Provides **control plane** connection
   - Ensures **reliability** and **mobility anchor**
   - Tracked in: `lte_cu_cp`, `lte_cu_up` tables

2. **gNB Secondary (cellId=2-8)**:
   - **Controlled by agent** (handover decisions)
   - Provides **high-throughput** data connection
   - Can be **handed over** between gNBs 2-8
   - Tracked in: `gnb_cu_cp`, `gnb_cu_up`, `du` tables

---

## Current Implementation Analysis

### What the Code Does:

1. **Reads UE features from `gnb_cu_cp` only:**
   ```python
   self.columns_state_ue = ["ue_x", "ue_y", 
                            "L3 serving SINR", "L3 serving Id(m_cellId)",  # From gnb_cu_cp
                            "DRB.UEThpDl.UEID", "DRB.BufferSize.Qos.UEID",  # From du
                            "L3 neigh SINR 1", "L3 neigh Id 1 (cellId)",   # From gnb_cu_cp
                            ...]
   ```
   - **`L3 serving Id(m_cellId)`** only exists in `gnb_cu_cp` table
   - This means the code only tracks the **gNB serving cell**, not LTE

2. **Filters out LTE from gNB list:**
   ```python
   # Keep only NR gNBs (labels 2–8 in enbs.txt), drop LTE anchor (1).
   gnb_ids = [cid for cid in sorted(gnb_dict.keys()) if cid >= 2][:n_gnbs]
   ```
   - Only considers gNBs 2-8 for handover decisions
   - LTE (cellId=1) is explicitly excluded

3. **Action controls only gNB handover:**
   ```python
   control_header=["timestamp", "ueId", "nrCellId"]  # nrCellId = NR (New Radio) cell ID
   ```
   - `nrCellId` means **NR cell ID** (gNBs 2-8), not LTE
   - Agent can only handover between gNBs, not to/from LTE

### What the Code Does NOT Do:

1. **Does NOT track LTE connection state:**
   - No features from `lte_cu_cp` table (e.g., `sameCellSinr`)
   - No LTE SINR in observations
   - No LTE throughput/buffer metrics

2. **Does NOT control LTE connection:**
   - Agent cannot handover to/from LTE
   - LTE connection is assumed to be always maintained

3. **Does NOT use LTE metrics in decision-making:**
   - Agent only sees gNB metrics
   - LTE anchor connection is "invisible" to the agent

---

## Implications

### ✅ What Works:
- **Multi-connectivity is active**: UEs maintain both LTE and gNB connections
- **gNB handover works**: Agent can handover between gNBs 2-8
- **Throughput optimization**: Agent can optimize gNB selection for better throughput

### ⚠️ What's Missing:
- **LTE metrics not used**: Agent doesn't see LTE SINR, throughput, or quality
- **LTE anchor not in graph**: Graph only includes gNBs 2-8, not LTE
- **No LTE-gNB coordination**: Agent doesn't know if LTE connection is good/bad

### 🤔 Potential Issues:
1. **Agent doesn't know LTE quality**: If LTE anchor has poor SINR, agent can't account for it
2. **Throughput might be combined**: `DRB.UEThpDl.UEID` might include both LTE and gNB throughput
3. **Buffer might be shared**: `DRB.BufferSize.Qos.UEID` might be for combined connection

---

## Recommendations

### If You Want to Include LTE in Observations:

1. **Add LTE features to UE observations:**
   ```python
   # Read from lte_cu_cp
   lte_sinr = datalake.read_kpms(timestamp, ["sameCellSinr"])
   # Add to ue_features: [x, y, gnb_sinr, lte_sinr, throughput, buffer, ...]
   ```

2. **Include LTE as a node in the graph:**
   ```python
   # Add LTE node (cellId=1) to graph
   # Create edges: UE ↔ LTE anchor
   # Edge attributes: [distance, is_lte_anchor, lte_sinr]
   ```

3. **Track LTE connection state:**
   - Monitor LTE SINR trends
   - Detect if LTE connection is degrading
   - Use LTE metrics in reward calculation

### Current State (Without Changes):
- **Multi-connectivity is active** in ns-3
- **Agent only controls gNB handover**
- **LTE connection is maintained but not observed/controlled**

---

## Verification Queries

To verify multi-connectivity yourself:

```sql
-- Check if all UEs have LTE connection
SELECT COUNT(DISTINCT ueImsiComplete) FROM lte_cu_cp 
WHERE timestamp = (SELECT MAX(timestamp) FROM lte_cu_cp);
-- Should return: 14 (all UEs)

-- Check if all UEs have gNB connection
SELECT COUNT(DISTINCT ueImsiComplete) FROM gnb_cu_cp 
WHERE timestamp = (SELECT MAX(timestamp) FROM gnb_cu_cp);
-- Should return: 14 (all UEs)

-- Check serving gNB cells (should be 2-8, never 1)
SELECT DISTINCT "L3 serving Id(m_cellId)" FROM gnb_cu_cp;
-- Should return: 2, 3, 4, 5, 6, 7, 8 (no 1)

-- Check LTE cellId (should always be 1)
SELECT DISTINCT cellId FROM lte_cu_cp;
-- Should return: 1
```

---

## Conclusion

**Answer to your question:**

✅ **YES, all UEs are connected to LTE (cellId=1) as an anchor**
✅ **YES, all UEs also connect to a gNB (cellId=2-8) for throughput**
✅ **This is multi-connectivity (EN-DC: E-UTRAN New Radio Dual Connectivity)**

**However:**
- The **agent only tracks and controls the gNB connection**
- The **LTE anchor connection is maintained but not observed/controlled by the agent**
- The **graph representation only includes gNBs 2-8, not LTE**

This is a **design choice**: The agent focuses on optimizing gNB handover for throughput, while LTE provides a stable anchor connection that doesn't need agent control.


# Key Performance Metrics (KPMs) Explanation

## Overview
This document explains all KPMs collected from ns-3 simulations in the HO-GRL environment, their sources, meanings, and importance for handover decision-making.

## Quick Reference: Answers to Common Questions

### Q1: Which metric gives throughput/data rate for each UE?
**Answer**: 
- **Primary**: `DRB.UEThpDl.UEID` from `du` table (bits per second)
- **Alternative**: `DRB.PdcpSduBitRateDl.UEID` from `lte_cu_up`/`gnb_cu_up` tables (PDCP layer throughput)
- **From volume**: `QosFlow.PdcpPduVolumeDL_Filter.UEID` from `du` table (calculate: volume / time_window)

### Q2: What does "L3" mean in "L3 serving SINR"?
**Answer**: 
- **L3 = Layer 3** = **RRC (Radio Resource Control) layer**
- L3 SINR is the **long-term averaged SINR** reported at the RRC layer
- Used for **handover decisions** (more stable than physical layer SINR)
- In 5G stack: L1=PHY, L2=MAC/RLC/PDCP, **L3=RRC**

### Q3: What is "average UE throughput at PDCP layer"?
**Answer**:
- Average data rate across all UEs, calculated from PDCP layer metrics
- Formula: `Sum(DRB.PdcpSduBitRateDl.UEID) / Number of Active UEs`
- Or: `Sum(QosFlow.PdcpPduVolumeDL_Filter.UEID) / (Time Window × Number of UEs)`
- **Why PDCP?** It's the highest L2 sublayer, represents actual user data rate before RLC segmentation

### Q4: How to compute latency for each UE?
**Answer**:
- **Method 1 (Recommended)**: `DRB.PdcpSduDelayDl.UEID (pdcpLatency)` from `lte_cu_up`/`gnb_cu_up` tables (already per-UE, in seconds)
- **Method 2**: Parse `DlE2PdcpStats.txt`, use `delay` column (11th column), group by IMSI
- **Method 3**: Calculate from packet traces: `latency = receive_time - transmit_time`

---

## 1. What is "L3" in "L3 serving SINR"?

**L3** refers to **Layer 3** in the OSI/5G protocol stack, specifically the **RRC (Radio Resource Control) layer**.

### 5G Protocol Stack Layers:
- **L1 (Physical Layer)**: Radio transmission, modulation, coding
- **L2 (Data Link Layer)**: 
  - **MAC (Medium Access Control)**: Scheduling, HARQ
  - **RLC (Radio Link Control)**: Segmentation, ARQ
  - **PDCP (Packet Data Convergence Protocol)**: Header compression, ciphering
- **L3 (Network Layer)**: 
  - **RRC (Radio Resource Control)**: Connection management, mobility, measurement reporting

**L3 serving SINR** means the SINR (Signal-to-Interference-plus-Noise Ratio) measured and reported at the RRC layer. This is the **long-term averaged SINR** used for handover decisions, as opposed to instantaneous physical layer SINR.

**Why L3 SINR?**
- More stable than L1 SINR (averaged over time)
- Used by RRC for mobility decisions (handover triggers)
- Represents the signal quality as perceived by the network layer

---

## 2. Throughput/Data Rate Metrics

### 2.1 Per-UE Throughput Metrics

#### **`DRB.UEThpDl.UEID`** (Primary metric for HO-GRL)
- **Source Table**: `du` (DU statistics)
- **Meaning**: **Downlink throughput per UE** (bits per second)
- **Layer**: Aggregated from PDCP/RLC/MAC layers
- **Usage**: 
  - Used in reward calculation (logarithmic throughput difference)
  - Primary performance indicator for each UE
  - Helps agent learn to maximize user data rates

#### **`DRB.UEThpDlPdcpBased.UEID`**
- **Source Table**: `du`
- **Meaning**: **PDCP-based downlink throughput per UE** (alternative calculation)
- **Layer**: PDCP layer only
- **Usage**: Alternative throughput metric, more accurate for PDCP layer performance

#### **`DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)`**
- **Source Table**: `lte_cu_up` or `gnb_cu_up`
- **Meaning**: **PDCP SDU bit rate** (throughput at PDCP layer)
- **Layer**: PDCP
- **Usage**: Measures actual data rate at PDCP layer before RLC segmentation

#### **`QosFlow.PdcpPduVolumeDL_Filter.UEID`**
- **Source Table**: `du` or `gnb_cu_up`
- **Meaning**: **PDCP PDU volume** (bytes transmitted)
- **Layer**: PDCP
- **Usage**: Total data volume, can be used to calculate throughput over time windows

### 2.2 Average UE Throughput at PDCP Layer

**Average UE throughput at PDCP layer** refers to the average data rate across all UEs, calculated from PDCP layer metrics. This is computed as:

```
Average Throughput = Sum(DRB.PdcpSduBitRateDl.UEID) / Number of Active UEs
```

Or from volume:
```
Average Throughput = Sum(QosFlow.PdcpPduVolumeDL_Filter.UEID) / (Time Window × Number of UEs)
```

**Why PDCP layer?**
- PDCP is the highest sublayer in L2
- Represents the actual user data rate (before RLC segmentation)
- More accurate than MAC layer throughput (which includes retransmissions)

---

## 3. Latency Metrics

### 3.1 Per-UE Latency

#### **`DRB.PdcpSduDelayDl.UEID (pdcpLatency)`**
- **Source Table**: `lte_cu_up` or `gnb_cu_up`
- **Meaning**: **PDCP SDU delay per UE** (seconds)
- **Calculation**: Time from PDCP SDU arrival to successful delivery
- **Includes**: 
  - Queueing delay at PDCP
  - RLC transmission delay
  - MAC scheduling delay
  - HARQ retransmission delay
- **Usage**: Critical for latency-sensitive applications (URLLC)

#### **`DRB.PdcpSduDelayDl(cellAverageLatency)`**
- **Source Table**: `lte_cu_up` or `gnb_cu_up`
- **Meaning**: **Average PDCP latency across all UEs in a cell**
- **Usage**: Cell-level latency metric for load balancing

### 3.2 E2 Interface Latency (from DlE2PdcpStats.txt)

From `DlE2PdcpStats.txt`:
- **`delay`**: End-to-end delay (seconds)
- **`stdDev`**: Standard deviation of delay
- **`min`**: Minimum delay
- **`max`**: Maximum delay

**Format**: `start | end | CellId | IMSI | RNTI | LCID | nTxPDUs | TxBytes | nRxPDUs | RxBytes | delay | stdDev | min | max | PduSize | ...`

**How to compute per-UE latency:**
```python
# From DlE2PdcpStats.txt
# delay column gives latency in seconds for each UE (identified by IMSI)
# This is the E2 interface measurement (more accurate than PDCP-only)
```

---

## 4. Database Tables and Their KPMs

### 4.1 `gnb_cu_cp` (gNB Control Plane - RRC Layer)

**Source Files**: `cu-cp-cell-*.txt` (cells 2-8)

**Key Metrics:**
- **`L3 serving SINR`**: RRC layer SINR for serving cell (dB)
- **`L3 serving Id(m_cellId)`**: Serving cell ID
- **`L3 neigh SINR 1-6`**: SINR for top 6 neighbor cells (dB)
- **`L3 neigh Id 1-6 (cellId)`**: Neighbor cell IDs
- **`numActiveUes`**: Number of active UEs in this cell
- **`DRB.EstabSucc.5QI.UEID (numDrb)`**: Number of established DRBs

**Why Important:**
- **Primary source for handover decisions**: L3 SINR is what triggers handovers
- **Neighbor cell information**: Essential for selecting target cells
- **Cell load**: `numActiveUes` indicates cell congestion

### 4.2 `lte_cu_cp` (LTE Anchor Control Plane)

**Source Files**: `cu-cp-cell-1.txt` (LTE anchor)

**Key Metrics:**
- **`sameCellSinr`**: SINR for LTE anchor cell
- Similar structure to `gnb_cu_cp` but for LTE

### 4.3 `gnb_cu_up` (gNB User Plane - PDCP Layer)

**Source Files**: `cu-up-cell-*.txt` (cells 2-8)

**Key Metrics:**
- **`QosFlow.PdcpPduVolumeDL_Filter.UEID(txPdcpPduBytesNrRlc)`**: PDCP PDU volume (bytes)
- **`DRB.PdcpPduNbrDl.Qos.UEID (txPdcpPduNrRlc)`**: Number of PDCP PDUs

**Why Important:**
- **Throughput calculation**: Volume metrics used to compute data rates
- **QoS flow tracking**: Per-QoS flow statistics

### 4.4 `lte_cu_up` (LTE Anchor User Plane)

**Source Files**: `cu-up-cell-1.txt`

**Key Metrics:**
- **`DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)`**: PDCP SDU volume (bytes)
- **`DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)`**: PDCP throughput (bps)
- **`DRB.PdcpSduDelayDl.UEID (pdcpLatency)`**: PDCP latency (seconds)
- **`Tot.PdcpSduNbrDl.UEID (txDlPackets)`**: Number of PDCP SDUs
- **`m_pDCPBytesDL(cellDlTxVolume)`**: Total cell downlink volume

**Why Important:**
- **Primary throughput source**: `DRB.PdcpSduBitRateDl.UEID` is the most accurate throughput metric
- **Latency measurement**: `DRB.PdcpSduDelayDl.UEID` for delay-sensitive applications
- **Cell-level aggregation**: `m_pDCPBytesDL` for cell load analysis

### 4.5 `du` (Distributed Unit - MAC/PHY Layer)

**Source Files**: `du-cell-*.txt` (cells 2-8)

**Key Metrics:**

#### Throughput & Volume:
- **`DRB.UEThpDl.UEID`**: **Primary throughput metric** (bps) ⭐
- **`DRB.UEThpDlPdcpBased.UEID`**: PDCP-based throughput (bps)
- **`QosFlow.PdcpPduVolumeDL_Filter.UEID`**: PDCP PDU volume (bytes)

#### Buffer & Queue:
- **`DRB.BufferSize.Qos.UEID`**: **Buffer size per UE** (bytes) ⭐
  - Indicates queued data waiting for transmission
  - High buffer = congestion or poor channel conditions

#### Resource Usage:
- **`RRU.PrbUsedDl`**: PRB (Physical Resource Block) usage per cell
- **`RRU.PrbUsedDl.UEID`**: PRB usage per UE
- **`dlAvailablePrbs`**: Available PRBs for scheduling
- **`ulAvailablePrbs`**: Available uplink PRBs

#### Transmission Blocks (TB):
- **`TB.TotNbrDl.1`**: Total number of downlink TBs (cell level)
- **`TB.TotNbrDl.1.UEID`**: Total number of downlink TBs (per UE)
- **`TB.TotNbrDlInitial`**: Initial transmission TBs (before retransmissions)
- **`TB.TotNbrDlInitial.Qpsk/16Qam/64Qam`**: TBs by modulation scheme
- **`TB.ErrTotalNbrDl.1`**: Erroneous TBs (cell level)
- **`TB.ErrTotalNbrDl.1.UEID`**: Erroneous TBs (per UE)

#### MCS (Modulation and Coding Scheme):
- **`CARR.PDSCHMCSDist.Bin1-6`**: MCS distribution bins (cell level)
- **`CARR.PDSCHMCSDist.Bin1-6.UEID`**: MCS distribution bins (per UE)

#### SINR Distribution:
- **`L1M.RS-SINR.Bin34/46/58/70/82/94/127`**: SINR distribution bins (cell level)
- **`L1M.RS-SINR.Bin*.UEID`**: SINR distribution bins (per UE)

#### Cell Load:
- **`DRB.MeanActiveUeDl`**: Mean number of active UEs in downlink

**Why Important:**
- **Most comprehensive table**: Contains throughput, buffer, resource usage, and quality metrics
- **Primary source for HO-GRL features**: Used for `DRB.UEThpDl.UEID` and `DRB.BufferSize.Qos.UEID`
- **Resource allocation**: PRB usage indicates cell load and scheduling efficiency

---

## 5. ns-3 Output Files and Their KPMs

### 5.1 E2 Interface Statistics (E2 = EPC to gNB interface)

#### **`DlE2PdcpStats.txt`** (Downlink E2 PDCP Statistics)
**Format**: `start | end | CellId | IMSI | RNTI | LCID | nTxPDUs | TxBytes | nRxPDUs | RxBytes | delay | stdDev | min | max | PduSize | ...`

**Metrics:**
- **`nTxPDUs`**: Number of transmitted PDUs
- **`TxBytes`**: Transmitted bytes
- **`nRxPDUs`**: Number of received PDUs
- **`RxBytes`**: Received bytes
- **`delay`**: **End-to-end delay** (seconds) ⭐
- **`stdDev`**: Delay standard deviation
- **`min/max`**: Min/max delay
- **`PduSize`**: Average PDU size

**Why Important:**
- **Most accurate latency measurement**: E2 interface captures true end-to-end delay
- **Throughput calculation**: `TxBytes / (end - start)` = throughput
- **Packet loss**: `nTxPDUs - nRxPDUs` indicates lost packets

#### **`DlE2PdcpStatsLte.txt`**
- Same as above but for LTE anchor (cellId = 1)

#### **`DlE2RlcStats.txt`** (Downlink E2 RLC Statistics)
**Format**: Similar to PDCP stats but for RLC layer

**Metrics:**
- RLC layer PDU statistics
- RLC-specific delays (includes ARQ retransmissions)

#### **`UlE2PdcpStats.txt`** / **`UlE2RlcStats.txt`**
- Uplink versions of the above

### 5.2 Handover Statistics

#### **`CellIdStats.txt`**
**Format**: `time | IMSI | cellId | handoverCount` (4 columns, space-separated)

**Metrics:**
- **`time`**: Timestamp (seconds)
- **`IMSI`**: UE identifier
- **`cellId`**: Current serving cell ID
- **`handoverCount`**: Cumulative number of handovers for this UE

**Why Important:**
- **Handover frequency tracking**: Counts how many times each UE has handed over
- **Cell association**: Shows which cell each UE is connected to at each time
- **Mobility pattern analysis**: Can identify ping-pong handovers
- **Real-time cell tracking**: Updates whenever UE changes cell

**Example from file:**
```
0.0542144 2 3 2
```
- Time: 0.0542144s
- UE 2 (IMSI=2) is connected to cell 3
- This UE has performed 2 handovers so far (cumulative count)

#### **`CellIdStatsHandover.txt`**
**Format**: `time | IMSI | sourceCellId | targetCellId` (4 columns, space-separated)

**Metrics:**
- **`time`**: Timestamp when handover occurred
- **`IMSI`**: UE identifier
- **`sourceCellId`**: Cell UE was leaving
- **`targetCellId`**: Cell UE is moving to

**Why Important:**
- **Handover event tracking**: Records each handover event with source/target
- **Handover success analysis**: Can identify failed handovers (if source != target)
- **Mobility trajectory**: Shows UE movement pattern
- **Ping-pong detection**: Identify UEs switching between same two cells

**Example from file:**
```
0.109268 4 3 3
```
- Time: 0.109268s
- UE 4 (IMSI=4) handed over from cell 3 to cell 3
- Note: Same source/target might indicate measurement report or handover preparation

**Another example:**
```
0.209268 7 2 8
```
- Time: 0.209268s
- UE 7 handed over from cell 2 to cell 8 (actual handover)

#### **`EnbHandoverEndStats.txt`** (eNB/gNB Side)
**Format**: `time | IMSI | sourceCellId | targetCellId | handoverCount` (5 columns, space-separated)

**Metrics:**
- **`time`**: Timestamp when handover completed
- **`IMSI`**: UE identifier
- **`sourceCellId`**: Source cell ID
- **`targetCellId`**: Target cell ID
- **`handoverCount`**: Sequential handover number

**Why Important:**
- **Network-side handover tracking**: Confirms handover completion from BS perspective
- **Handover duration**: Can calculate time between start and end
- **Handover success verification**: Network confirms successful handover

**Example from file:**
```
0.109768 10 3 3
```
- Time: 0.109768s
- UE 10 completed handover from cell 3 to cell 3
- Handover count: 3 (this is the 3rd handover for this UE)

#### **`EnbHandoverStartStats.txt`**
**Format**: Same as `EnbHandoverEndStats.txt`
- Records when handover procedure **starts** (from BS perspective)
- Can calculate handover duration: `end_time - start_time`

#### **`UeHandoverStartStats.txt`** / **`UeHandoverEndStats.txt`**
**Format**: `time | IMSI | sourceCellId | targetCellId | handoverCount` (5 columns)
- UE-side handover statistics (from UE perspective)
- Same format as eNB stats but from UE's point of view
- Useful for comparing network vs UE perspective timing

### 5.3 Physical Layer Statistics

#### **`DlMacStats.txt`** (Downlink MAC Statistics)
**Format**: `time | cellId | IMSI | frame | sframe | RNTI | mcsTb1 | sizeTb1 | mcsTb2 | sizeTb2 | ccId`

**Metrics:**
- **`mcsTb1/mcsTb2`**: MCS (Modulation and Coding Scheme) for transport blocks
- **`sizeTb1/sizeTb2`**: Transport block size (bits)
- **`frame/sframe`**: Frame and subframe numbers

**Why Important:**
- **Scheduling information**: Shows what MCS and TB size were allocated
- **Link adaptation**: MCS indicates channel quality adaptation
- **Throughput calculation**: `sizeTb / time_interval` = instantaneous throughput

#### **`DlRxPhyStats.txt`** / **`DlTxPhyStats.txt`**
- Physical layer receive/transmit statistics
- SINR, power levels, interference

#### **`DlPhyTransmissionTrace.txt`**
**Format**: `DL/UL | time | frame | subF | slot | 1stSym | symbol# | cellId | rnti | ccId | tbSize | mcs | rv | SINR(dB) | corrupt | TBler`

**Metrics:**
- **`SINR(dB)`**: Instantaneous SINR per transmission
- **`corrupt`**: Whether TB was corrupted (0/1)
- **`TBler`**: Transport Block Error Rate
- **`mcs`**: MCS used
- **`tbSize`**: Transport block size

**Why Important:**
- **Per-transmission SINR**: Most granular SINR measurement
- **Error rate**: `TBler` shows transmission reliability
- **Link quality**: Instantaneous channel conditions

### 5.4 Protocol Layer Statistics

#### **`DlPdcpStats.txt`** / **`UlPdcpStats.txt`**
- PDCP layer statistics (similar to E2 but from protocol stack)

#### **`DlRlcStats.txt`** / **`UlRlcStats.txt`**
- RLC layer statistics
- ARQ retransmissions, segmentation

#### **`RlcAmBufferSize.txt`**
- RLC AM (Acknowledged Mode) buffer size
- Shows queued data at RLC layer

### 5.5 SINR and Signal Quality

#### **`LteDlRsrpSinrStats.txt`**
- LTE downlink RSRP (Reference Signal Received Power) and SINR
- For LTE anchor cell

#### **`LteUlSinrStats.txt`** / **`LteUlInterferenceStats.txt`**
- LTE uplink SINR and interference

#### **`MmWaveSinrTime.txt`**
- mmWave SINR over time
- For mmWave gNBs (cells 2-8)

#### **`MmWaveSwitchStats.txt`**
- mmWave beam switching statistics

### 5.6 Other Statistics

#### **`RxPacketTrace.txt`**
- Packet-level trace
- Shows each packet's journey through the stack

#### **`EnbSchedAllocTraces.txt`**
- Scheduler allocation traces
- Shows which UEs were scheduled in each TTI

#### **`X2Stats.txt`**
- X2 interface statistics (inter-eNB communication)
- Handover signaling

#### **`UeFailures.txt`**
- UE failure events (RLF, connection drops)

---

## 6. Summary: Key Metrics for HO-GRL

### 6.1 Throughput/Data Rate
| Metric | Table | Description | Usage |
|--------|-------|-------------|-------|
| `DRB.UEThpDl.UEID` | `du` | **Primary throughput** (bps) | Reward calculation, performance |
| `DRB.PdcpSduBitRateDl.UEID` | `lte_cu_up` | PDCP throughput (bps) | Alternative throughput |
| `QosFlow.PdcpPduVolumeDL_Filter.UEID` | `du` | PDCP volume (bytes) | Throughput calculation |

### 6.2 Latency
| Metric | Table | Description | Usage |
|--------|-------|-------------|-------|
| `DRB.PdcpSduDelayDl.UEID` | `lte_cu_up` | PDCP latency (seconds) | Latency-sensitive apps |
| `delay` (from DlE2PdcpStats.txt) | File | E2 end-to-end delay | Most accurate latency |

### 6.3 Signal Quality
| Metric | Table | Description | Usage |
|--------|-------|-------------|-------|
| `L3 serving SINR` | `gnb_cu_cp` | **RRC layer SINR** (dB) | Handover decisions |
| `L3 neigh SINR 1-6` | `gnb_cu_cp` | Neighbor SINRs (dB) | Target cell selection |

### 6.4 Buffer/Queue
| Metric | Table | Description | Usage |
|--------|-------|-------------|-------|
| `DRB.BufferSize.Qos.UEID` | `du` | **Buffer size** (bytes) | Congestion indicator |

### 6.5 Resource Usage
| Metric | Table | Description | Usage |
|--------|-------|-------------|-------|
| `RRU.PrbUsedDl` | `du` | PRB usage (%) | Cell load |
| `numActiveUes` | `gnb_cu_cp` | Active UEs per cell | Cell congestion |

### 6.6 Handover Tracking
| File | Description | Usage |
|------|-------------|-------|
| `CellIdStats.txt` | Current cell per UE | Cell association |
| `CellIdStatsHandover.txt` | Handover events | Handover frequency |
| `EnbHandoverEndStats.txt` | HO completion | HO success rate |

---

## 7. How to Compute Latency for Each UE

### Method 1: From Database (Recommended)
```python
# Read from lte_cu_up or gnb_cu_up table
latency = datalake.read_kpms(timestamp, ["DRB.PdcpSduDelayDl.UEID (pdcpLatency)"])
# Returns: [(ueImsiComplete, latency_value), ...]
```

### Method 2: From DlE2PdcpStats.txt
```python
# Parse the file
# delay column (11th column) gives latency in seconds
# Group by IMSI to get per-UE latency
```

### Method 3: Calculate from Packet Traces
```python
# From RxPacketTrace.txt or similar
# latency = receive_time - transmit_time
# Average over time window for per-UE latency
```

**Recommended**: Use `DRB.PdcpSduDelayDl.UEID` from database as it's already aggregated and per-UE.

---

## 8. File-to-Table Mapping

| ns-3 File | Database Table | Key Metrics |
|-----------|----------------|-------------|
| `cu-cp-cell-*.txt` | `gnb_cu_cp` (cells 2-8) or `lte_cu_cp` (cell 1) | L3 SINR, neighbor cells, numActiveUes |
| `cu-up-cell-*.txt` | `gnb_cu_up` (cells 2-8) or `lte_cu_up` (cell 1) | PDCP throughput, latency, volume |
| `du-cell-*.txt` | `du` (cells 2-8) | Throughput, buffer, PRB usage, TB stats |
| `DlE2PdcpStats.txt` | Not in DB (raw file) | E2 delay, packet stats |
| `CellIdStats.txt` | Not in DB (raw file) | Cell association, HO count |
| `CellIdStatsHandover.txt` | Not in DB (raw file) | HO events (source/target) |

---

## 9. Necessity of Each KPM for HO-GRL

### Critical (Must Have):
1. **`L3 serving SINR`**: Primary signal quality metric for handover decisions
2. **`DRB.UEThpDl.UEID`**: Performance metric, used in reward
3. **`DRB.BufferSize.Qos.UEID`**: Congestion indicator
4. **`L3 neigh SINR 1-3`**: Target cell selection
5. **`numActiveUes`**: Cell load balancing

### Important (Should Have):
6. **`RRU.PrbUsedDl`**: Resource utilization
7. **`DRB.PdcpSduDelayDl.UEID`**: Latency for QoS
8. **Handover statistics**: For analysis and optimization

### Optional (Nice to Have):
9. **MCS distribution**: Link adaptation quality
10. **TB error rates**: Transmission reliability
11. **E2 interface stats**: Detailed latency analysis

---

## 10. Example: Reading Metrics

```python
# Throughput per UE
throughput = datalake.read_kpms(timestamp, ["DRB.UEThpDl.UEID"])
# Returns: [(ueImsiComplete, throughput_bps), ...]

# Latency per UE
latency = datalake.read_kpms(timestamp, ["DRB.PdcpSduDelayDl.UEID (pdcpLatency)"])
# Returns: [(ueImsiComplete, latency_seconds), ...]

# SINR and serving cell
sinr_data = datalake.read_kpms(timestamp, ["L3 serving SINR", "L3 serving Id(m_cellId)"])
# Returns: [(ueImsiComplete, sinr_dB, cellId), ...]

# Buffer size
buffer = datalake.read_kpms(timestamp, ["DRB.BufferSize.Qos.UEID"])
# Returns: [(ueImsiComplete, buffer_bytes), ...]
```

---

This document provides a comprehensive overview of all KPMs. For specific use cases, refer to the relevant sections above.


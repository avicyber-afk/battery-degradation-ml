# Pull the per-cycle discharge curves (Qdlin, dQ/dV) out of the raw .mat files
# and turn them into features. Writes two tables:
#   MATR_augmented.csv       - per cycle: summary scalars + curve-shape features
#   MATR_cell_dQ_features.csv - per cell: Severson dQ(V) = Qdlin[100] - Qdlin[10]
# The 4 batch-1 cells that spill into batch 2 are spliced back (same fix as
# BatteryML); the 5 batch-2 continuation cells aren't emitted on their own.

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

RAW = Path(__file__).parent
FILES = [
    ('batch1', RAW / 'MATR_batch_20170512.mat'),
    ('batch2', RAW / 'MATR_batch_20170630.mat'),
    ('batch3', RAW / 'MATR_batch_20180412.mat'),
    ('batch4', RAW / 'MATR_batch_20190124.mat'),
]
OUT_PERCYCLE = RAW / 'MATR_augmented.csv'
OUT_PERCELL = RAW / 'MATR_cell_dQ_features.csv'

# batch1 cells that continue as these batch2 cells (BatteryML clean_batches)
CONT_B1 = ['c0', 'c1', 'c2', 'c3', 'c4']
CONT_B2 = ['c7', 'c8', 'c9', 'c15', 'c16']

V_TARGETS = [2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.3]
EARLY_REF, LATE_REF = 10, 100        # reference cycles for dQ(V)


def get_arr(f, ref):
    return np.hstack(f[ref][:]).astype('float64')


def vgrid_indices(vgrid):
    return {vt: int(np.argmin(np.abs(vgrid - vt))) for vt in V_TARGETS}


def cycle_features(Qdlin, dQdV, vgrid, vidx):
    f = {}
    q = Qdlin
    f['Qdlin_var'] = float(np.nanvar(q))
    f['Qdlin_min'] = float(np.nanmin(q))
    f['Qdlin_max'] = float(np.nanmax(q))
    f['Qdlin_range'] = f['Qdlin_max'] - f['Qdlin_min']
    for vt, idx in vidx.items():
        f[f'Qdlin_at_{vt}V'] = float(q[idx])
    if np.all(np.isnan(dQdV)):
        f['dQdV_min'] = np.nan; f['dQdV_peak_V'] = np.nan; f['dQdV_var'] = np.nan
    else:
        f['dQdV_min'] = float(np.nanmin(dQdV))                 # main IC peak (negative)
        f['dQdV_peak_V'] = float(vgrid[int(np.nanargmin(dQdV))])
        f['dQdV_var'] = float(np.nanvar(dQdV))
    return f


def dq_features(q_early, q_late):
    """Severson delta-Q(V) statistics between two early cycles."""
    dq = q_late - q_early
    eps = 1e-12
    return {
        'dQ_var_log': float(np.log10(np.nanvar(dq) + eps)),
        'dQ_min_log': float(np.log10(np.abs(np.nanmin(dq)) + eps)),
        'dQ_mean_log': float(np.log10(np.abs(np.nanmean(dq)) + eps)),
        'dQ_skew': float(skew(dq, nan_policy='omit')),
        'dQ_kurt': float(kurtosis(dq, nan_policy='omit')),
        'dQ_at_2V_log': float(np.log10(np.abs(dq[0]) + eps)),
    }


def process_cell(f, batch, cyc_group, summ_group, cyc_life, vgrid, vidx,
                 cell_key, start_cycle=0, id_override=None):
    """Yield per-cycle feature rows for one cell; also return the Qdlin arrays
    captured at the early/late reference cycles for the dQ(V) features."""
    n = cyc_group['V'].shape[0]
    IR = np.hstack(summ_group['IR'][0, :].tolist())
    QD = np.hstack(summ_group['QDischarge'][0, :].tolist())
    QC = np.hstack(summ_group['QCharge'][0, :].tolist())
    TA = np.hstack(summ_group['Tavg'][0, :].tolist())
    TX = np.hstack(summ_group['Tmax'][0, :].tolist())
    TM = np.hstack(summ_group['Tmin'][0, :].tolist())
    CT = np.hstack(summ_group['chargetime'][0, :].tolist())
    bid = id_override or f'MATR_{batch}_{cell_key}'

    rows = []
    q_ref = {}
    for j in range(n):
        cyc_no = start_cycle + j
        if cyc_no == 0:
            continue                          # skip formation cycle
        try:
            Qdlin = get_arr(f, cyc_group['Qdlin'][j, 0])
            dQdV = get_arr(f, cyc_group['discharge_dQdV'][j, 0])
        except Exception:
            continue
        row = {'battery_id': bid, 'batch': batch, 'test_id': cyc_no}
        # summary scalars (guard against index drift)
        row['Capacity'] = float(QD[j]) if j < len(QD) else np.nan
        row['Re'] = float(IR[j]) if j < len(IR) else np.nan
        row['QCharge'] = float(QC[j]) if j < len(QC) else np.nan
        row['ambient_temperature'] = float(TA[j]) if j < len(TA) else np.nan
        row['Tmax'] = float(TX[j]) if j < len(TX) else np.nan
        row['Tmin'] = float(TM[j]) if j < len(TM) else np.nan
        row['chargetime'] = float(CT[j]) if j < len(CT) else np.nan
        row['cycle_life'] = float(cyc_life)
        row.update(cycle_features(Qdlin, dQdV, vgrid, vidx))
        rows.append(row)
        if cyc_no in (EARLY_REF, LATE_REF):
            q_ref[cyc_no] = Qdlin
    return rows, q_ref, bid


def main():
    # ---- pass 1: batch2 to grab continuation cells' data ----
    print('Pass 1: reading batch2 continuation cells ...')
    f2 = h5py.File(FILES[1][1], 'r')
    b2 = f2['batch']
    vgrid2 = np.hstack(f2[b2['Vdlin'][0, 0]][:])     # per-cell ref; grid is fixed
    vidx2 = vgrid_indices(vgrid2)
    cont_rows = {}          # batch2 cell_key -> (rows, q_ref, n_cycles)
    for k in CONT_B2:
        i = int(k[1:])
        cl = float(np.array(f2[b2['cycle_life'][i, 0]]).flatten()[0])
        rows, q_ref, _ = process_cell(f2, 'batch2', f2[b2['cycles'][i, 0]],
                                      f2[b2['summary'][i, 0]], cl, vgrid2, vidx2, k)
        cont_rows[k] = (rows, q_ref, len(rows))
    # keep f2 open; we still need it for the non-continuation batch2 cells

    all_rows = []
    cell_feats = []

    for batch, path in FILES:
        print(f'Processing {batch}: {path.name}')
        f = f2 if batch == 'batch2' else h5py.File(path, 'r')
        b = f['batch']
        vgrid = np.hstack(f[b['Vdlin'][0, 0]][:])
        vidx = vgrid_indices(vgrid)
        n_cells = b['summary'].shape[0]

        for i in range(n_cells):
            k = f'c{i}'
            if batch == 'batch2' and k in CONT_B2:
                continue                      # emitted as part of batch1 cell
            cl = float(np.array(f[b['cycle_life'][i, 0]]).flatten()[0])
            rows, q_ref, bid = process_cell(f, batch, f[b['cycles'][i, 0]],
                                            f[b['summary'][i, 0]], cl, vgrid, vidx, k)
            # splice: append the matching batch2 continuation cell
            if batch == 'batch1' and k in CONT_B1:
                b2k = CONT_B2[CONT_B1.index(k)]
                cont, cont_qref, _ = cont_rows[b2k]
                offset = max((r['test_id'] for r in rows), default=0)
                for r in cont:
                    r2 = dict(r); r2['battery_id'] = bid; r2['batch'] = batch
                    r2['test_id'] = r['test_id'] + offset
                    r2['cycle_life'] = float(cl)
                    all_rows.append(r2)
            all_rows.extend(rows)

            # per-cell dQ(V) features from early reference cycles
            qe = q_ref.get(EARLY_REF)
            ql = q_ref.get(LATE_REF)
            cf = {'battery_id': bid, 'batch': batch, 'cycle_life': cl}
            if qe is not None and ql is not None and len(qe) == len(ql):
                cf.update(dq_features(qe, ql))
            cell_feats.append(cf)

            if (i + 1) % 10 == 0:
                print(f'  {batch} cell {i+1}/{n_cells}  (rows so far {len(all_rows)})')
        if batch != 'batch2':
            f.close()
    f2.close()

    df = pd.DataFrame(all_rows)
    df = df[df['test_id'] > 0].reset_index(drop=True)
    df.to_csv(OUT_PERCYCLE, index=False)
    print(f'\nPer-cycle augmented table: {df.shape} -> {OUT_PERCYCLE.name}')

    cf = pd.DataFrame(cell_feats)
    cf.to_csv(OUT_PERCELL, index=False)
    print(f'Per-cell dQ(V) table: {cf.shape} ({cf["dQ_var_log"].notna().sum()} cells with dQ features)'
          f' -> {OUT_PERCELL.name}')


if __name__ == '__main__':
    main()

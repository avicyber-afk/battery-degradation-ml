# Flatten the raw MATR .mat files into one per-cycle CSV. Just the cycle-level
# summaries here (capacity, IR, temps, charge time) - the full V/I/T curves are
# handled in battery_curve_features.py.
# Columns: battery_id, batch, test_id, ambient_temperature (Tavg),
# Capacity (QDischarge), Re (IR), chargetime, cycle_life. MATR has one IR per
# cycle, so no separate Rct.

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent
BATCH_FILES = [
    ('batch1', RAW_DIR / 'MATR_batch_20170512.mat'),
    ('batch2', RAW_DIR / 'MATR_batch_20170630.mat'),
    ('batch3', RAW_DIR / 'MATR_batch_20180412.mat'),
    ('batch4', RAW_DIR / 'MATR_batch_20190124.mat'),
]
OUT_CSV = RAW_DIR / 'MATR_battery_summary.csv'


def load_batch_summary(file):
    """Extract only the lightweight per-cycle summary fields for every cell
    in a batch (no full V/I/T/Qd waveforms)."""
    f = h5py.File(file, 'r')
    batch = f['batch']
    num_cells = batch['summary'].shape[0]
    cells = {}
    for i in range(num_cells):
        cycle_life = float(np.array(f[batch['cycle_life'][i, 0]]).flatten()[0])
        summary_grp = f[batch['summary'][i, 0]]
        IR = np.hstack(summary_grp['IR'][0, :].tolist())
        QD = np.hstack(summary_grp['QDischarge'][0, :].tolist())
        TA = np.hstack(summary_grp['Tavg'][0, :].tolist())
        CT = np.hstack(summary_grp['chargetime'][0, :].tolist())
        CY = np.hstack(summary_grp['cycle'][0, :].tolist())
        cells[f'c{i}'] = dict(cycle_life=cycle_life, IR=IR, QD=QD, TA=TA, CT=CT, CY=CY)
    f.close()
    return cells


def main():
    print('Loading batch summaries (BatteryML MATR field extraction)...')
    batches = {}
    for name, path in BATCH_FILES:
        print(f'  {name}: {path.name}')
        batches[name] = load_batch_summary(path)
        print(f'    -> {len(batches[name])} cells')

    # Same fix as BatteryML's clean_batches(): 4 cells from batch1 continue
    # into batch2 under different indices; splice them back onto batch1 and
    # drop the batch2 duplicates.
    batch2_keys = ['c7', 'c8', 'c9', 'c15', 'c16']
    batch1_keys = ['c0', 'c1', 'c2', 'c3', 'c4']
    for b1k, b2k in zip(batch1_keys, batch2_keys):
        b1 = batches['batch1'][b1k]
        b2 = batches['batch2'][b2k]
        offset = len(b1['CY'])
        b1['IR'] = np.hstack([b1['IR'], b2['IR']])
        b1['QD'] = np.hstack([b1['QD'], b2['QD']])
        b1['TA'] = np.hstack([b1['TA'], b2['TA']])
        b1['CT'] = np.hstack([b1['CT'], b2['CT']])
        b1['CY'] = np.hstack([b1['CY'], b2['CY'] + offset])
        b1['cycle_life'] = b1['cycle_life'] + b2['cycle_life']
    for b2k in batch2_keys:
        del batches['batch2'][b2k]

    rows = []
    for batch_name, cells in batches.items():
        for cell_key, d in cells.items():
            battery_id = f'MATR_{batch_name}_{cell_key}'
            n = len(d['QD'])
            for j in range(n):
                rows.append((
                    battery_id, batch_name, int(d['CY'][j]) if j < len(d['CY']) else j + 1,
                    float(d['TA'][j]) if j < len(d['TA']) else np.nan,
                    float(d['QD'][j]),
                    float(d['IR'][j]) if j < len(d['IR']) else np.nan,
                    float(d['CT'][j]) if j < len(d['CT']) else np.nan,
                    float(d['cycle_life']),
                ))

    df = pd.DataFrame(rows, columns=[
        'battery_id', 'batch', 'test_id', 'ambient_temperature',
        'Capacity', 'Re', 'chargetime', 'cycle_life'
    ])
    # Cycle 0 is a diagnostic/formation cycle in the MATR protocol, not a
    # regular aging cycle -- drop it, matching common practice for this dataset.
    df = df[df['test_id'] > 0].reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print(f'\nWrote {len(df):,} rows across {df.battery_id.nunique()} cells to {OUT_CSV}')
    print(df.describe())


if __name__ == '__main__':
    main()

# Reads the 2000 episode CSVs once into one compressed cache, so training loads quickly.
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data_randomized"  # folder with episode_XXXX.csv
CACHE = Path(__file__).parent / "data_randomized_cache.npz"  # output file

def load_csv(file):
    # One episode: table, column names and a nav_done flag per row.
    data = np.genfromtxt(file, delimiter=",", names=True)
    columns = data.dtype.names
    table = np.column_stack([data[c] for c in columns])
    wheels = np.column_stack([data[f"wheel_cmds_{i}"] for i in range(4)])
    moving = np.any(wheels !=0, axis=1)
    last = np.where(moving)[0][-1]  # last row where the wheels still move
    nav_done = np.arange(len(table)) > last

    return table, columns, nav_done

def main():
    files = sorted(DATA_DIR.glob("episode_[0-9][0-9][0-9][0-9].csv"))
    tables, episodes, dones = [], [], []
    for file in files:
        table, columns, nav_done = load_csv(file)
        episode = int(file.stem.split("_")[1])
        tables.append(table)
        episodes.append(np.full(len(table), episode))
        dones.append(nav_done)
        if episode%100 == 0:
            print(f"Episode :{episode}")
    np.savez_compressed(
        CACHE,
        table=np.concatenate(tables),
        columns=np.array(columns),
        episode=np.concatenate(episodes),
        nav_done=np.concatenate(dones)
    )
    print(f"Saved {len(files)} episodes, {sum(len(t) for t in tables)} rows to {CACHE}")


if __name__ == "__main__":
    main()

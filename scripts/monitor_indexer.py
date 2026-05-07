import subprocess, sys

result = subprocess.run(['pgrep', '-f', 'core.indexer'], capture_output=True, text=True)
if result.returncode == 0:
    print("STATUS: RUNNING")
    sys.exit(0)

# Process finished — read log tail
log = subprocess.run(
    ['tail', '-30', '/mnt/e/engineer-companion/_indexer_run.log'],
    capture_output=True, text=True
)
print("STATUS: DONE")
print(log.stdout)

# Check DB exists
from pathlib import Path
db = Path('/mnt/e/engineer-companion/assets/db/engineer.db')
if db.exists():
    import lancedb
    dbc = lancedb.connect(str(db))
    if 'chunks' in dbc.table_names():
        tbl = dbc.open_table('chunks')
        print(f"DB: {tbl.count_rows()} rows")
    else:
        print("DB: no chunks table")
else:
    print("DB: not found")

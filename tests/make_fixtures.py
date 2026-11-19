"""Build small FineVideo-like Parquet files (mp4 + json columns) with good and damaged videos.

    python tests/make_fixtures.py <videos dir> <out dir>
"""
import json, os, subprocess, sys
import pyarrow as pa, pyarrow.parquet as pq

vids, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rd = lambda n: open(os.path.join(vids, n), "rb").read()
schema = pa.schema([("mp4", pa.binary()), ("json", pa.string())])

def write(name, rows, row_group_size=3):
    t = pa.table({"mp4": [r for r in rows], "json": [json.dumps({"i": i}) for i in range(len(rows))]}, schema=schema)
    pq.write_table(t, os.path.join(out, name), row_group_size=row_group_size)

# faststart copy so that a truncated file still opens and fails part-way
fs = os.path.join(out, "_fast.mp4")
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(vids, "v_zidane_720p30.mp4"), "-c", "copy",
                "-movflags", "+faststart", fs], check=True)
fast = open(fs, "rb").read(); os.remove(fs)
z = rd("v_zidane_720p30.mp4")
rows0 = [z, rd("v_bus_24.mp4"), rd("v_bus_rot.mp4"), None, b"this is not an mp4" * 100,
         z[: len(z) // 2], fast[: int(len(fast) * 0.6)], rd("v_zid_rot270.mp4"), rd("v_zid_rot180.mp4"),
         rd("v_bus_2fps.mp4"), b""]
write("train-00000-of-01357.parquet", rows0)
# unreadable parquet: valid file cut in half
write("tmp.parquet", rows0[:3]); b = open(os.path.join(out, "tmp.parquet"), "rb").read(); os.remove(os.path.join(out, "tmp.parquet"))
open(os.path.join(out, "train-00001-of-01357.parquet"), "wb").write(b[: len(b) // 2])
# a file whose second row group is damaged (bytes overwritten inside its data pages)
write("train-00002-of-01357.parquet", [rd("v_bus_2fps.mp4"), rd("v_zid_rot180.mp4")], row_group_size=1)
p = os.path.join(out, "train-00002-of-01357.parquet"); md = pq.ParquetFile(p).metadata
rg1 = md.row_group(1).column(0); size = rg1.total_compressed_size
start = rg1.dictionary_page_offset if rg1.has_dictionary_page else rg1.data_page_offset
data = bytearray(open(p, "rb").read()); data[start + 4: start + size - 4] = os.urandom(size - 8); open(p, "wb").write(data)
# no mp4 column
pq.write_table(pa.table({"json": ["{}", "{}"]}), os.path.join(out, "train-00003-of-01357.parquet"))
print("fixtures written to", out)

"""Build the exact v1 encoder input text for every test record (row order = our rid order) and a
5,000-record training check set with the stored vectors."""
import numpy as np, polars as pl

V1 = "query: name: {} | address: {} | country: {}"

def text(df):
    return df.select(text=pl.format(V1, pl.col("name_norm").fill_null(""), pl.col("addr_norm").fill_null(""),
                                    pl.col("country").str.to_lowercase()))

def main():
    te = pl.read_parquet("Embeddings/test_records.parquet", columns=["entity_id", "name_norm", "addr_norm", "country"])
    ours = pl.read_parquet("work/norm/test.parquet", columns=["entity_id"])["entity_id"]
    assert te["entity_id"].equals(ours), "test record order differs from our rid order"
    text(te).write_parquet("work/remote_upload/test_text.parquet", compression="zstd")
    tr = pl.read_parquet("Embeddings/train_records.parquet", columns=["name_norm", "addr_norm", "country"])
    idx = np.sort(np.random.default_rng(7).choice(tr.height, 5000, replace=False))
    text(tr[idx]).write_parquet("work/remote_upload/train_check_text.parquet")
    np.save("work/remote_upload/train_check_emb.npy", np.asarray(np.load("Embeddings/train_emb.npy", mmap_mode="r")[idx]))
    print(f"test texts: {te.height:,}; e.g. {text(te[:2])['text'].to_list()}")

if __name__ == "__main__":
    main()

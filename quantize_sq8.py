import faiss
import shutil
import os

src = r"c:\LBSrv\pp-mi-asistents-prasmes\knowledge_base"
dst = r"c:\LBSrv\pp-mi-asistents-prasmes\knowledge_base_sq8"

index_names = ["faiss_esco_en", "faiss_skillsfuture_idx", "faiss_vas_kompetences"]

for name in index_names:
    dst_dir = os.path.join(dst, name)
    os.makedirs(dst_dir, exist_ok=True)

    src_index_path = os.path.join(src, name, "index.faiss")
    src_pkl_path   = os.path.join(src, name, "index.pkl")
    dst_index_path = os.path.join(dst_dir, "index.faiss")
    dst_pkl_path   = os.path.join(dst_dir, "index.pkl")

    print(f"\n=== {name} ===")
    flat_idx = faiss.read_index(src_index_path)
    n, d = flat_idx.ntotal, flat_idx.d
    print(f"  Vectors: {n:,}  Dimensions: {d}")

    # Build SQ8 index — train on up to 50k vectors
    sq_idx = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
    train_n = min(50_000, n)
    print(f"  Training SQ8 on {train_n:,} vectors ...")
    sq_idx.train(flat_idx.reconstruct_n(0, train_n))

    # Add all vectors in batches to keep memory usage reasonable
    BATCH = 10_000
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        sq_idx.add(flat_idx.reconstruct_n(start, end - start))
        print(f"  Added {end:,}/{n:,}", end="\r")
    print()

    faiss.write_index(sq_idx, dst_index_path)
    shutil.copy2(src_pkl_path, dst_pkl_path)

    orig_mb = os.path.getsize(src_index_path) / 1_048_576
    new_mb  = os.path.getsize(dst_index_path) / 1_048_576
    print(f"  index.faiss: {orig_mb:.1f} MB → {new_mb:.1f} MB  ({new_mb/orig_mb*100:.1f}% of original)")

print("\nDone!")

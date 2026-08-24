import os
import glob


def test_load_model_from_ckpt():
    # Test menemukan ckpt di models/version_0/checkpoints
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Source Code Model",
        "models",
    )
    version_dir = os.path.join(models_dir, "version_0")
    ckpt_dir = os.path.join(version_dir, "checkpoints")

    # Cek struktur folder
    assert os.path.isdir(models_dir), f"Models dir tidak ada: {models_dir}"
    assert os.path.isdir(version_dir), f"Version dir tidak ada: {version_dir}"
    assert os.path.isdir(ckpt_dir), f"Checkpoints dir tidak ada: {ckpt_dir}"

    # Cari ckpt files
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    assert len(ckpt_files) >= 1, "Tidak ada file .ckpt ditemukan"

    # Cek training_dataset.pkl juga ada
    pkl_files = glob.glob(os.path.join(ckpt_dir, "*.pkl"))
    assert len(pkl_files) >= 1, "Tidak ada file .pkl ditemukan"

    # Ambil checkpoint yang pertama
    ckpt_path = ckpt_files[0]
    assert os.path.isfile(ckpt_path), f"File ckpt tidak valid: {ckpt_path}"

    print(f"✓ Checkpoint ditemukan: {ckpt_path}")
    print(f"✓ Size: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")
    print(f"✓ Dataset pkl: {pkl_files[0]}")

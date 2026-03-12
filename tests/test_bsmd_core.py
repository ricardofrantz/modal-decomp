import numpy as np
import pytest
from pymodal import BSMDAnalyzer


def _make_analyzer(tmp_path, triads, nfft=4, Ns=10, Nspace=4):
    """Helper to build a BSMDAnalyzer with synthetic data."""
    Nx = int(np.sqrt(Nspace))
    Ny = Nspace // Nx
    data = {
        'q': np.random.randn(Ns, Nspace),
        'x': np.linspace(0, 1, Nx),
        'y': np.linspace(0, 1, Ny),
        'dt': 1.0,
        'Nx': Nx,
        'Ny': Ny,
        'Ns': Ns,
    }
    analyzer = BSMDAnalyzer(
        file_path='dummy.h5',
        nfft=nfft,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
        use_static_triads=True,
        static_triads=triads,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    return analyzer


def test_static_bsmd_core_small(tmp_path):
    """Basic smoke test: single zero-frequency triad produces results."""
    analyzer = _make_analyzer(tmp_path, triads=[(0, 0, 0)])
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    assert analyzer.modes1.shape[0] == 1
    assert analyzer.modes1.shape[1] == 4


def test_negative_frequency_conjugate_symmetry(tmp_path):
    """Negative frequency bin indices are served via conjugate symmetry."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    # qhat has shape (nfft//2+1, Nspace, Nblocks) = (5, 4, Nblocks)
    assert analyzer.qhat.shape[0] == 5  # bins 0..4

    # Directly check the helper: qhat[-1] should equal conj(qhat[1])
    q_pos = analyzer._get_qhat_for_index(1)
    q_neg = analyzer._get_qhat_for_index(-1)
    np.testing.assert_array_equal(q_neg, np.conj(q_pos))

    # Run BSMD — should not crash with negative indices
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    assert not np.isnan(analyzer.eigenvalues[0])


def test_out_of_range_index_skipped(tmp_path):
    """Triads referencing bins beyond qhat range produce NaN, not a crash."""
    analyzer = _make_analyzer(tmp_path, triads=[(99, -99, 0)], nfft=4, Ns=10)
    # nfft=4 → only 3 frequency bins (0,1,2). Index 99 is out of range.
    analyzer._perform_static_bsmd_core()
    assert np.isnan(np.abs(analyzer.eigenvalues[0]))


def test_triadic_constraint_violation_skipped(tmp_path):
    """Triads that violate p1+p2=p3 are skipped with NaN eigenvalue."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 1)], nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    assert np.isnan(np.abs(analyzer.eigenvalues[0]))


def test_multiple_triads_with_negatives(tmp_path):
    """Multiple triads including negative bins all produce finite results."""
    triads = [(1, -1, 0), (2, -2, 0), (1, 1, 2), (0, 0, 0)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (len(triads),)
    assert analyzer.modes1.shape == (len(triads), 4)
    assert analyzer.modes2.shape == (len(triads), 4)
    # All valid triads should produce finite eigenvalues
    for idx, (p1, p2, p3) in enumerate(triads):
        assert not np.isnan(analyzer.eigenvalues[idx]), f"Triad {(p1,p2,p3)} produced NaN"


def test_bispectral_correlation_uses_all_three_frequencies(tmp_path):
    """Verify that the bispectral correlation C involves Q1, Q2, AND Q3.

    Construct a case where Q3 is zeroed out.  If the algorithm correctly
    uses Q3 as B in C = A^H W B, all eigenvalues should be zero.
    """
    triads = [(1, 1, 2)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24)
    # Zero out qhat at bin 2 (= p3) → B = 0 → C = 0 → eigenvalue = 0
    analyzer.qhat[2, :, :] = 0.0
    analyzer._perform_static_bsmd_core()
    assert np.abs(analyzer.eigenvalues[0]) == pytest.approx(0.0, abs=1e-12)


def test_disk_backed_qhat_matches_ram(tmp_path):
    """Disk-backed mode (max_qhat_gb=0) produces identical results to RAM mode."""
    triads = [(1, -1, 0), (2, -2, 0), (1, 1, 2), (0, 0, 0)]
    np.random.seed(42)

    # RAM mode (default)
    ram = _make_analyzer(tmp_path / "ram", triads=triads, nfft=8, Ns=24)
    ram._perform_static_bsmd_core()

    # Disk-backed mode: max_qhat_gb=0 forces offload on any qhat
    np.random.seed(42)
    Nspace = 4
    Nx, Ny = 2, 2
    data = {
        'q': np.random.randn(24, Nspace),
        'x': np.linspace(0, 1, Nx),
        'y': np.linspace(0, 1, Ny),
        'dt': 1.0, 'Nx': Nx, 'Ny': Ny, 'Ns': 24,
    }
    disk_dir = tmp_path / "disk"
    disk_dir.mkdir()
    disk = BSMDAnalyzer(
        file_path='dummy.h5', nfft=8, overlap=0.0,
        results_dir=disk_dir, figures_dir=disk_dir,
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
        use_static_triads=True, static_triads=triads,
        max_qhat_gb=0,  # force disk-backed
    )
    disk.load_and_preprocess()
    disk.compute_fft_blocks()
    assert disk._qhat_on_disk, "Expected disk-backed mode with max_qhat_gb=0"

    disk._perform_static_bsmd_core()

    np.testing.assert_allclose(np.abs(disk.eigenvalues), np.abs(ram.eigenvalues), rtol=1e-12)
    np.testing.assert_allclose(np.abs(disk.modes1), np.abs(ram.modes1), rtol=1e-12)
    np.testing.assert_allclose(np.abs(disk.modes2), np.abs(ram.modes2), rtol=1e-12)
    disk.close()

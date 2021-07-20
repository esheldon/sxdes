import pytest
import numpy as np
import sxdes


def _make_image(rng):
    dims = 32, 32
    sigma = 2.0
    counts = 100.0
    noise = 0.1

    cen = (np.array(dims)-1)/2
    cen += rng.uniform(low=-2, high=2, size=2)

    rows, cols = np.mgrid[
        0: dims[0],
        0: dims[1],
    ]

    rows = rows - cen[0]
    cols = cols - cen[1]

    norm = 1.0/(2 * np.pi * sigma**2)

    image = np.exp(-0.5*(rows**2 + cols**2)/sigma**2)
    image *= counts*norm

    # import hickory
    # plt = hickory.Plot(aratio=1)
    # plt.imshow(image)
    # plt.show()
    return image, noise, cen


def test_detect_smoke():
    rng = np.random.RandomState(646509750)
    image, noise, _ = _make_image(rng)
    _ = sxdes.run_sep(image, noise)


def test_detect():
    rng = np.random.RandomState(60970)

    for i in range(10):
        image, noise, cen = _make_image(rng)
        cat, seg = sxdes.run_sep(image, noise)
        assert cat.size > 0

        s = cat['flux'].argsort()[-1]
        row, col = cat['y'][s], cat['x'][s]
        assert abs(row - cen[0]) < 1
        assert abs(col - cen[1]) < 1


def test_mask():
    rng = np.random.RandomState(60970)

    for i in range(10):
        image, noise, cen = _make_image(rng)
        mask = np.ones(image.shape, dtype=bool)
        cat, seg = sxdes.run_sep(image, noise, mask=mask)
        assert cat.size == 0


def test_errors():
    rng = np.random.RandomState(60970)

    image, noise, cen = _make_image(rng)
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        cat, seg = sxdes.run_sep(image, noise, mask=mask)


def test_seg():
    rng = np.random.RandomState(60970)

    for i in range(10):
        image, noise, cen = _make_image(rng)
        cat, seg = sxdes.run_sep(image, noise)
        assert cat.size > 0

        for i in range(1, cat.size+1):
            msk = seg == i
            assert cat["isoarea_image"][i-1] == np.sum(msk)

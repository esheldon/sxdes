"""
TODO

    - fluxerr_auto always comes back zero
    - add windowed quantities

"""
import numpy as np
import sep

# used half light radius
PHOT_FLUXFRAC = 0.5

DETECT_THRESH = 0.8
SX_CONFIG = {

    'deblend_cont': 0.00001,

    'deblend_nthresh': 64,

    'minarea': 4,

    'filter_type': 'conv',

    # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
    'filter_kernel':  np.array([
        [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
        [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
        [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
        [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],
        [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
        [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
        [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
    ]),
}


def run_sep(image, noise, config=SX_CONFIG, thresh=DETECT_THRESH):
    """
    Run sep on the image using DES parameters

    Parameters
    ----------
    image: array
        The image to extract
    noise: float or array
        A representation of the noise in the image
    config: dict, optional
        Config parameters for the sep run. Default are the DES Y6 settings,
        which can be found in sxdes.SX_CONFIG
    thresh: float, optional
        The threshold for detection, default sxdes.DETECT_THRESH

    Returns
    -------
    cat, seg:
        The catalog and seg map
    """

    runner = SepRunner(image, noise, config=config, thresh=thresh)
    return runner.cat, runner.seg


class SepRunner(object):
    """
    Parameters
    ----------
    image: array
        The image to extract
    noise: float or array
        A representation of the noise in the image
    config: dict, optional
        Config parameters for the sep run. Default are the DES Y6 settings,
        which can be found in sxdes.SX_CONFIG
    thresh: float, optional
        The threshold for detection, default sxdes.DETECT_THRESH

    The resulting catalog and seg map can be gotten through the .cat and .seg
    attributes
    """
    def __init__(self, image, noise, config=SX_CONFIG, thresh=DETECT_THRESH):
        self.image = image
        self.noise = noise
        self.config = config.copy()
        self.thresh = thresh

        self._run_sep()

    @property
    def cat(self):
        """
        get a reference to the catalog
        """
        return self._cat

    @property
    def seg(self):
        """
        get a reference to the seg map
        """
        return self._seg

    def _run_sep(self):
        objs, seg = sep.extract(
            self.image,
            self.thresh,
            err=self.noise,
            segmentation_map=True,
            **self.config
        )

        auto_res = self._get_flux_auto(objs)

        cat = self._add_fields(objs, seg, auto_res)

        self._seg = seg
        self._cat = cat

    def _add_fields(self, objs, seg, auto_res):
        flux_auto, fluxerr_auto, flux_radius, kron_radius = auto_res

        new_dt = [
            ('number', 'i4'),
            ('kron_radius', 'f4'),
            ('flux_auto', 'f4'),
            ('fluxerr_auto', 'f4'),
            ('flux_radius', 'f4'),
            ('isoarea_image', 'f4'),
            ('iso_radius', 'f4'),
        ]

        all_dt = objs.dtype.descr + new_dt
        cat = np.zeros(objs.size, dtype=all_dt)

        for d in objs.dtype.descr:
            name = d[0]
            cat[name] = objs[name]

        cat['number'] = np.arange(1, cat.size+1)
        cat['kron_radius'] = kron_radius
        cat['flux_auto'] = flux_auto
        cat['fluxerr_auto'] = fluxerr_auto
        cat['flux_radius'] = flux_radius

        # use the number of pixels in the seg map as the iso area
        for i in range(objs.size):
            w = np.where(seg == (i+1))
            cat['isoarea_image'][i] = w[0].size

        cat['iso_radius'] = np.sqrt(cat['isoarea_image'].clip(min=1)/np.pi)
        return cat

    def _get_flux_auto(self, objs):
        flux_auto = np.zeros(objs.size)-9999.0
        fluxerr_auto = np.zeros(objs.size)-9999.0
        flux_radius = np.zeros(objs.size)-9999.0
        kron_radius = np.zeros(objs.size)-9999.0

        w, = np.where(
            (objs['a'] >= 0.0) & (objs['b'] >= 0.0) &
            (objs['theta'] >= -np.pi/2.) & (objs['theta'] <= np.pi/2.)
        )

        if w.size > 0:
            kron_radius[w], krflag = sep.kron_radius(
                self.image,
                objs['x'][w],
                objs['y'][w],
                objs['a'][w],
                objs['b'][w],
                objs['theta'][w],
                6.0,
            )
            objs['flag'][w] |= krflag

            aper_rad = 2.5*kron_radius
            flux_auto[w], fluxerr_auto[w], flag_auto = \
                sep.sum_ellipse(
                    self.image,
                    objs['x'][w],
                    objs['y'][w],
                    objs['a'][w],
                    objs['b'][w],
                    objs['theta'][w],
                    aper_rad[w],
                    subpix=1,
                )
            objs['flag'][w] |= flag_auto

            flux_radius[w], frflag = sep.flux_radius(
                self.image,
                objs['x'][w],
                objs['y'][w],
                6.*objs['a'][w],
                PHOT_FLUXFRAC,
                normflux=flux_auto[w],
                subpix=5,
            )
            objs['flag'][w] |= frflag  # combine flags into 'flag'

        return flux_auto, fluxerr_auto, flux_radius, kron_radius

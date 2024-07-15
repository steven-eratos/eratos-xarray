import unittest
import os
from eratos.creds import AccessTokenCreds
import xarray as xr


class EratosSDKITest(unittest.TestCase):

    def setUp(self) -> None:
        eratos_id = os.getenv("ERATOS_ID")
        eratos_secret = os.getenv("ERATOS_SECRET")

        if not eratos_id or not eratos_secret:
            raise RuntimeError('Unable to run test. Please configure ERATOS_ID and ERATOS_SECRET environment variables.')

        self.ecreds = AccessTokenCreds(
            eratos_id,
            eratos_secret
        )

    def test_silo(self):
        """
        Open remote SILO dataset, slice to known spatial area.
        :return:
        """
        # Import the backend module to ensure it is registered
        import eratos_xarray

        silo = xr.open_dataset('ern:e-pn.io:resource:eratos.blocks.silo.maxtemperature', eratos_auth=self.ecreds)
        bars_silo = silo.sel(dict(lat=slice(-34.91, -34.02), lon=slice(148.03, 148.97))).sel(time=slice("2023-06-02", "2023-06-10"))
        self.assertEqual((9, 18, 19), bars_silo['max_temp'].shape)

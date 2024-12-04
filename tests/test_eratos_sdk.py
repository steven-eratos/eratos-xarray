import unittest
import os
import datetime
from eratos.creds import AccessTokenCreds
from eratos.errors import CommError
import xarray as xr
import numpy as np


class EratosSDKITest(unittest.TestCase):
    def setUp(self) -> None:
        eratos_id = os.getenv("ERATOS_ID")
        eratos_secret = os.getenv("ERATOS_SECRET")

        if not eratos_id or not eratos_secret:
            raise RuntimeError(
                "Unable to run test. Please configure ERATOS_ID and ERATOS_SECRET environment variables."
            )

        self.ecreds = AccessTokenCreds(eratos_id, eratos_secret)

    def test_silo(self):
        """
        Open remote SILO dataset, slice to known spatial area.
        :return:
        """
        # Import the backend module to ensure it is registered
        import eratos_xarray

        silo = xr.open_dataset(
            "ern:e-pn.io:resource:eratos.blocks.silo.maxtemperature",
            eratos_auth=self.ecreds,
        )
        bars_silo = silo.sel(
            dict(lat=slice(-34.91, -34.02), lon=slice(148.03, 148.97))
        ).sel(time=slice("2023-06-02", "2023-06-10"))

        self.assertEqual(bars_silo["lat"][0], -34.9)
        self.assertEqual(bars_silo["lat"][-1], -34.05)
        self.assertEqual(bars_silo["lon"][0], 148.05)
        self.assertEqual(bars_silo["lon"][-1], 148.95)
        self.assertEqual(
            bars_silo["time"].data[0], np.datetime64("2023-06-02T00:00:00.000000000")
        )
        self.assertEqual(
            bars_silo["time"].data[-1], np.datetime64("2023-06-10T00:00:00.000000000")
        )
        self.assertEqual((9, 18, 19), bars_silo["max_temp"].shape)

    def test_silo_time_drill(self):
        """
        Open remote SILO dataset, slice to known spatial area.
        :return:
        """
        # Import the backend module to ensure it is registered
        import eratos_xarray

        silo = xr.open_dataset(
            "ern:e-pn.io:resource:eratos.blocks.silo.maxtemperature",
            eratos_auth=self.ecreds,
        )
        bars_silo = (
            silo.sel(dict(lat=-34.91, lon=148.03), method="nearest")
            .sel(time=slice("2023-06-02", "2023-06-10"))
            .load()
        )

        self.assertEqual(bars_silo["lat"], -34.9)
        self.assertEqual(bars_silo["lat"].shape, ())
        self.assertEqual(bars_silo["lon"], 148.05)
        self.assertEqual(bars_silo["lon"].shape, ())
        self.assertEqual(
            bars_silo["time"].data[0], np.datetime64("2023-06-02T00:00:00.000000000")
        )
        self.assertEqual(
            bars_silo["time"].data[-1], np.datetime64("2023-06-10T00:00:00.000000000")
        )
        self.assertEqual(xr.core.formatting.first_n_items(bars_silo.max_temp, 1), 18.4)
        self.assertEqual(bars_silo["max_temp"].shape, (9,))

    def test_silo_time_drill_no_import(self):
        """
        Open remote SILO dataset, slice to known spatial area, do not load eratos_xarray directly
        :return:
        """

        silo = xr.open_dataset(
            "ern:e-pn.io:resource:eratos.blocks.silo.maxtemperature",
            engine="eratos",
            eratos_auth=self.ecreds,
        )
        bars_silo = (
            silo.sel(dict(lat=-34.91, lon=148.03), method="nearest")
            .sel(time=slice("2023-06-02", "2023-06-10"))
            .load()
        )

        self.assertEqual(bars_silo["lat"], -34.9)
        self.assertEqual(bars_silo["lat"].shape, ())
        self.assertEqual(bars_silo["lon"], 148.05)
        self.assertEqual(bars_silo["lon"].shape, ())
        self.assertEqual(
            bars_silo["time"].data[0], np.datetime64("2023-06-02T00:00:00.000000000")
        )
        self.assertEqual(
            bars_silo["time"].data[-1], np.datetime64("2023-06-10T00:00:00.000000000")
        )
        self.assertEqual(xr.core.formatting.first_n_items(bars_silo.max_temp, 1), 18.4)

    def test_silo_out_of_bounds_indexing(self):
        """
        Open remote SILO dataset, slice out of bounds
        :return:
        """

        today = datetime.datetime.today()

        silo = xr.open_dataset(
            "ern:e-pn.io:resource:eratos.blocks.silo.maxtemperature",
            eratos_auth=self.ecreds,
        )
        silo_bad_start = silo.sel(
            time=slice(
                today + datetime.timedelta(days=1), today + datetime.timedelta(days=2)
            )
        )
        silo_bad_end = silo.sel(time=slice("1880-01-01", "1880-02-01"))

        try:
            silo_bad_start = silo_bad_start.load()
            silo_bad_end = silo_bad_end.load()
        except CommError as e:
            self.fail("Did not slice and load successfully")

        self.assertEqual(silo_bad_start.max_temp.size, 0)
        self.assertEqual(silo_bad_end.max_temp.size, 0)

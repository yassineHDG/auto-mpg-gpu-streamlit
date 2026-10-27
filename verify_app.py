"""Vérifications exécutables sans pytest : python verify_app.py.

Les doublures CUDA vérifient le routage et les erreurs, pas les kernels GPU.
Le test matériel reste à réaliser sur la machine NVIDIA.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

import app


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.data, cls.audit = app.load_data()
        cls.bundle = app.train_model(cls.raw, force_cpu=True)

    def test_data_and_units(self):
        self.assertEqual(len(self.raw), 398)
        self.assertEqual(self.raw.horsepower.isna().sum(), 6)
        self.assertFalse(self.data[app.FEATURES].isna().any().any())
        np.testing.assert_allclose(self.data.l_100km, 235.215 / self.data.mpg)
        np.testing.assert_allclose(self.data.co2_g_km, self.data.l_100km * 23.5)
        self.assertAlmostEqual(10 * 23.5 * 15000 / 1e6, 3.525)
        self.assertEqual(self.data.year.min(), 1970)
        self.assertEqual(self.data.year.max(), 1982)

    def test_preprocessing_has_no_test_leakage(self):
        train, test = train_test_split(self.raw, test_size=0.2, random_state=app.SEED)
        b = self.bundle
        self.assertEqual((b.train_n, b.test_n), (318, 80))
        np.testing.assert_allclose(
            b.preprocessor.named_transformers_["numeric"].statistics_,
            train[app.NUMERIC_FEATURES].median().to_numpy(),
        )
        self.assertEqual(set(b.preprocessor.feature_names_in_), set(app.FEATURES))
        self.assertNotIn("mpg", b.preprocessor.feature_names_in_)
        self.assertNotIn("co2_g_km", b.preprocessor.feature_names_in_)
        np.testing.assert_allclose(b.X_test[:, -3:].sum(axis=1), 1)
        np.testing.assert_array_equal(b.test_rows.vehicle_id, test.vehicle_id)
        self.assertGreater(b.r2, 0.80)
        self.assertLess(b.rmse, b.baseline_rmse)

    def test_filters_and_small_sample(self):
        self.assertTrue(app.filter_data(self.data, [], (1970, 1982), [4], "").empty)
        self.assertTrue(app.filter_data(self.data, app.ORIGINS, (1970, 1982), [], "").empty)
        self.assertTrue(app.filter_data(self.data, app.ORIGINS, (1970, 1982), [4], "[.*").empty)
        toyota = app.filter_data(self.data, app.ORIGINS, (1970, 1982), [3, 4, 5, 6, 8], "TOYOTA")
        self.assertEqual(len(toyota), 25)
        with self.assertRaises(ValueError):
            app.train_model(self.raw.head(5), force_cpu=True)

    def test_prediction_and_importance(self):
        b = self.bundle
        example = self.raw[app.FEATURES].iloc[[0]]
        X = b.preprocessor.transform(example)
        self.assertGreater(app.predict_encoded(b, X)[0], 0)
        importance = app.permutation_importance_gpu_aware(b)
        self.assertEqual(len(importance), 7)
        self.assertTrue(np.isfinite(importance["Hausse de RMSE (MPG)"]).all())

    def test_no_cuda_routes_to_cpu(self):
        with patch.object(app, "get_cuda", return_value=(None, "", ["CUDA absent simulé"])):
            b = app.train_model(self.raw, n_estimators=10)
        self.assertEqual(b.backend, "sklearn")
        self.assertFalse(b.is_gpu)
        self.assertIn("CUDA absent simulé", b.notices)

    def test_gpu_training_failure_and_silent_cpu_fallback(self):
        # CuPy et moteurs factices : ne représentent PAS une exécution GPU.
        fake_cp = SimpleNamespace(
            asarray=np.asarray,
            cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)),
        )

        class BrokenCuML:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                raise RuntimeError("Erreur CUDA simulée pendant fit")

        class SilentCpuXGB:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                return self

            def get_booster(self):
                return self

            def save_config(self):
                return json.dumps({"learner": {"generic_param": {"device": "cpu"}}})

        cuml = ModuleType("cuml")
        ensemble = ModuleType("cuml.ensemble")
        ensemble.RandomForestRegressor = BrokenCuML
        cuml.ensemble = ensemble
        xgb = ModuleType("xgboost")
        xgb.__version__ = "2.1.4"
        xgb.XGBRegressor = SilentCpuXGB
        with (
            patch.dict(sys.modules, {"cuml": cuml, "cuml.ensemble": ensemble, "xgboost": xgb}),
            patch.object(app, "get_cuda", return_value=(fake_cp, "GPU simulé", [])),
        ):
            b = app.train_model(self.raw, n_estimators=10)
        self.assertEqual(b.backend, "sklearn")
        self.assertTrue(any("Erreur CUDA simulée" in n for n in b.notices))
        self.assertTrue(any("basculé sur cpu" in n for n in b.notices))

    def test_streamlit_interactions_and_model_reuse(self):
        at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.tabs), 4)
        first = at.session_state["model_bundle"]
        at.button(key="FormSubmitter:vehicle_prediction-Prédire la consommation").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertIs(at.session_state["model_bundle"], first)
        result = at.session_state["vehicle_result"]
        self.assertGreater(result["mpg"], 0)
        # Le catalogue du comparateur reste disponible, même sans résultat filtré.
        at.sidebar.multiselect[0].set_value([]).run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Aucun véhicule" in x.value for x in at.info))
        self.assertEqual(len(at.selectbox(key="compare_a").options), 398)
        # Retirer un filtre vide permet de réutiliser le modèle pour les mêmes données.
        at.sidebar.multiselect[0].set_value(app.ORIGINS).run()
        self.assertEqual(len(at.exception), 0)
        at.sidebar.slider[1].set_value(20).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["model_bundle"].model.n_estimators, 20)
        at.radio[1].set_value("Cylindres").run()
        at.selectbox(key="compare_a").select_index(10).run()
        self.assertEqual(len(at.exception), 0)

    def test_runtime_gpu_failure_recomputes_prediction_on_cpu(self):
        at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60).run()
        original = at.session_state["model_bundle"]
        # Remplace uniquement le moteur stocké par un modèle GPU qui lève une erreur.
        class BrokenPrediction:
            def predict(self, X):
                raise RuntimeError("Panne GPU simulée")

        fake_cp = SimpleNamespace(
            asarray=np.asarray,
            cuda=SimpleNamespace(Device=lambda i: SimpleNamespace(use=lambda: None)),
        )
        at.session_state["model_bundle"] = replace(
            original, backend="cuml", cp=fake_cp, gpu_name="GPU simulé", model=BrokenPrediction()
        )
        at.button(key="FormSubmitter:vehicle_prediction-Prédire la consommation").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["model_bundle"].backend, "sklearn")
        self.assertGreater(at.session_state["vehicle_result"]["mpg"], 0)
        self.assertTrue(any("Panne GPU simulée" in n for n in at.session_state["model_bundle"].notices))


if __name__ == "__main__":
    unittest.main(verbosity=2)

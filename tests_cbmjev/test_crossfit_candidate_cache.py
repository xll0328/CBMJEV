import copy
import unittest
from unittest.mock import patch

from cbmjev.contracts import Concept, QueryGroup, Schema, candidate_actions, stable_hash
from cbmjev.crossfit_training import _candidate_pool, _validate_target_package
from cbmjev.learning import normalize_config
from tests_cbmjev import test_crossfit_training as fixtures


class CandidateCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NestedOOFTrainingPrimitiveTests.setUpClass()
        cls.f = fixtures.NestedOOFTrainingPrimitiveTests

    def test_reuses_candidates_without_skipping_record_validation(self):
        cfg = normalize_config(fixtures.CONFIG, self.f.schema)
        package = self.f.package
        with patch("cbmjev.crossfit_training.candidate_actions", wraps=candidate_actions) as calls:
            provenance = _validate_target_package(package, self.f.schema, cfg)
        masks = {tuple(v >= 0 for v in r["observed"]) for r in package["records"]}
        self.assertEqual(calls.call_count, len(masks))
        self.assertLess(calls.call_count, package["record_count"])
        with patch("cbmjev.crossfit_training._candidate_pool",
                   side_effect=lambda schema, config: lambda mask: frozenset(candidate_actions(
                       tuple(0 if visible else -1 for visible in mask), schema,
                       include_pairs=config["include_pairs"], include_all=config["include_all"],
                       pairs=config["pairs"]))) :
            uncached = _validate_target_package(package, self.f.schema, cfg)
        self.assertEqual(provenance, uncached)

    def test_warm_cache_cannot_hide_bad_categories_or_actions(self):
        cfg = normalize_config(fixtures.CONFIG, self.f.schema)
        # Put a valid event before the bad event, then recompute transport hashes:
        # rejection must come from full semantic validation, not stale hashes.
        for field, value in (("observed", [True, -1, -1]),
                             ("observed", [999, -1, -1]),
                             ("action", [True]), ("action", [0, 0]),
                             ("action", [999]), ("action", [0])):
            package = copy.deepcopy(self.f.package)
            for record in package["records"][:2]:
                record["observed"] = [0, -1, -1]
                record["action"] = []
            package["records"][1][field] = value
            package["records_sha256"] = stable_hash(package["records"])
            package.pop("package_sha256")
            package["package_sha256"] = stable_hash(package)
            with patch("cbmjev.crossfit_training.candidate_actions", wraps=candidate_actions) as calls:
                with self.assertRaisesRegex(ValueError, "invalid observed|invalid acquisition|reacquire"):
                    _validate_target_package(package, self.f.schema, cfg)
                self.assertGreaterEqual(calls.call_count, 1)

    def test_pool_bound_eviction_and_scope(self):
        schema = Schema("cache-bound-fixture", 2,
            tuple(Concept(str(i), str(i), ("n", "y")) for i in range(12)),
            tuple(QueryGroup(str(i), (i,)) for i in range(12)))
        cfg = normalize_config({"device": "cpu"}, schema)
        pool = _candidate_pool(schema, cfg)
        empty = (False,) * 12
        for code in range(4096):
            pool(tuple(bool(code & (1 << i)) for i in range(12)))
        self.assertEqual(pool.cache_info().currsize, 2048)
        self.assertEqual(pool.cache_info().misses, 4096)
        pool(empty)
        self.assertEqual(pool.cache_info().misses, 4097)
        self.assertEqual(_candidate_pool(schema, cfg).cache_info().currsize, 0)

"""Portable read-only bundle verification on tiny temporary fixtures."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "cbmjev_test_bundle_verifier", Path(__file__).resolve().parents[1] / "tools/verify_bundle.py")
_VERIFIER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VERIFIER)
verify = _VERIFIER.verify


class BundleVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cbmjev-bundle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "moved-project"
        self.root.mkdir()
        self.old = self.base / "old-machine" / "project"

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_explicit_relocation_checks_current_target_without_rewriting(self):
        self.write("docs/target note.md", "Existing target.\n")
        document = self.write("review.md", "[historical](<" + str(self.old) +
                              "/docs/target%20note.md:42#section>)\n")
        original_contents = document.read_bytes()
        unrelocated = verify(self.root)
        self.assertEqual(unrelocated["status"], "FAIL")
        report = verify(self.root, original_root=self.old)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["relocated_links"], 1)
        self.assertEqual(len(report["relocated_references"]), 1)
        self.assertEqual(report["relocated_references"][0]["relocated_target"],
                         str((self.root / "docs/target note.md").resolve()))
        self.assertEqual(document.read_bytes(), original_contents)
        self.assertFalse(self.old.exists())

    def test_unrelated_and_prefix_collision_absolute_links_still_fail(self):
        self.write("target.md", "Must not rescue unrelated references.\n")
        unrelated = self.base / "other-project" / "target.md"
        same_prefix = self.old.parent / "project-other" / "target.md"
        self.write("review.md", "[outside]({})\n[prefix]({})\n".format(unrelated, same_prefix))
        report = verify(self.root, original_root=self.old)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["relocated_links"], 0)
        self.assertEqual({row["missing_link"] for row in report["failures"]},
                         {str(unrelated), str(same_prefix)})

    def test_parent_traversal_outside_original_root_is_not_relocated(self):
        self.write("target.md", "Existing migrated file.\n")
        outside = str(self.old) + "/../target.md"
        self.write("review.md", "[outside]({})\n".format(outside))
        report = verify(self.root, original_root=self.old)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["relocated_links"], 0)

    def test_excluded_environment_and_build_trees_are_pruned(self):
        self.write("valid.json", '{"ok": true}')
        for directory in (".venv", "build", "__pycache__", ".git", "nested/.venv"):
            self.write(directory + "/deep/invalid.json", "not valid JSON")
            self.write(directory + "/broken.md", "[missing](does-not-exist)")
        report = verify(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["json"], 1)
        self.assertEqual(report["counts"]["local_links"], 0)

    def test_relative_links_jsonl_and_svg_keep_existing_behavior(self):
        self.write("docs/target.md", "Target.\n")
        self.write("docs/source.md", "[ok](target.md#anchor)\n[web](https://example.com)\n"
                   "```\n[ignored](missing-in-code)\n```\n")
        self.write("rows.jsonl", '{"a": 1}\n\n{"b": 2}\n')
        self.write("figure.svg", '<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        report = verify(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["local_links"], 1)
        self.assertEqual(report["counts"]["jsonl_rows"], 2)
        self.assertEqual(report["counts"]["svg"], 1)

    def test_missing_relocated_target_fails_and_relative_original_root_is_rejected(self):
        self.write("review.md", "[missing]({})\n".format(self.old / "missing.md"))
        report = verify(self.root, original_root=self.old)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["relocated_links"], 1)
        with self.assertRaisesRegex(ValueError, "explicit absolute"):
            verify(self.root, original_root="relative/project")


if __name__ == "__main__":
    unittest.main()

"""CPU-only checks for the independent training/evaluation entry points."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComponentLayoutTests(unittest.TestCase):
    def test_sft_cli_does_not_require_verl_or_gpu_libraries(self):
        script = (
            "import sys; from methods.sft import train; "
            "assert not {'torch', 'transformers', 'peft', 'verl'} & set(sys.modules); "
            "sys.argv = ['sft', '--help']; train.main()"
        )
        run = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        self.assertIn('--model-path', run.stdout)
        self.assertIn('--train-file', run.stdout)

    def test_evaluation_inspection_works_outside_checkout_without_gpu_imports(self):
        script = (
            "import sys; from monitorability_eval.cli import main; main(['inspect']); "
            "assert not {'torch', 'vllm', 'transformers', 'verl'} & set(sys.modules)"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'evaluation' / 'src'))
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, '-c', script], cwd=tmp, env=env,
                                 capture_output=True, text=True, check=True)
        report = json.loads(run.stdout)
        self.assertEqual(report['integrity']['datasets'], 12)
        self.assertEqual(len(report['profiles']['task_models']), 3)
        self.assertEqual(len(report['profiles']['monitors']), 3)

    def test_root_package_discovery_excludes_evaluation(self):
        tree = ast.parse((ROOT / 'setup.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'find_packages']
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
        self.assertEqual(keywords['include'], ['verl', 'verl.*', 'methods', 'methods.*'])

    def test_only_dataset_jsonl_has_ignore_exception(self):
        rules = (ROOT / '.gitignore').read_text().splitlines()
        self.assertIn('*.jsonl', rules)
        self.assertEqual([r for r in rules if r.startswith('!') and 'jsonl' in r],
                         ['!/evaluation/src/monitorability_eval/assets/datasets/**/*.jsonl'])


if __name__ == '__main__':
    unittest.main()

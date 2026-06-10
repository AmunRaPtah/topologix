"""Scaffold sanity tests. Real tests land with each issue."""
import topologix


def test_version():
    assert topologix.__version__ == "0.0.1"


def test_modules_importable():
    from topologix import complex, metric, homology, features, screen, llm, wolfram, data  # noqa: F401

"""Module entrypoint.

Allows running with: `python -m emotion_nlp --data-dir data`
"""

from .train import main

if __name__ == "__main__":
    main()


import os

from hypothesis import settings

# "ci" keeps the suite well under a minute on a busy machine; "long" is for deeper runs:
#   HYPOTHESIS_PROFILE=long pytest tests/test_properties.py tests/test_fuzz.py
settings.register_profile("ci", max_examples=30, deadline=None)
settings.register_profile("long", max_examples=5000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

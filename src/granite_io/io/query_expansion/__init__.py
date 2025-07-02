# SPDX-License-Identifier: Apache-2.0


# Local
from .query_expansion import QueryExpansionIOProcessor

# Expose public symbols at `granite_io.io.query_expansion` to save users from typing
__all__ = [
    "QueryExpansionIOProcessor",
]
